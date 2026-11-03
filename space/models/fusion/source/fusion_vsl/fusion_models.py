"""M08 dual-branch Functional classifiers, no training or encoder dependencies.

M07 layers/helpers are reused without changing their implementations or APIs.
Each branch gets NEW layer instances and independent variables. This module is
imported lazily by models.py's M08 facade to avoid an eager circular import.
"""
from dataclasses import dataclass
import json
import math
from numbers import Real
from pathlib import Path
import unicodedata

import tensorflow as tf

from .config import atomic_write_text, sha256_file
from .feature_cache import canonical, fingerprint, is_sha
from .models import MaskedTemporalAttention, ValidateTemporalInput, baseline_inputs, _temporal_checks

VERSION = "m08_dual_branch_v1"
INPUT_CONTRACT = {"rgb_features": (16,1280), "landmark_features": (16,300), "valid_mask": (16,)}


@dataclass(frozen=True)
class FusionModelConfig:
    projection_units: int = 128
    gru_units: int = 64
    attention_units: int = 64
    classifier_units: int = 128
    dropout: float = 0.4

    def __post_init__(self):
        for name in ("projection_units", "gru_units", "attention_units", "classifier_units"):
            if type(getattr(self,name)) is not int or getattr(self,name)<1:
                raise ValueError(f"M08: {name} must be a positive integer")
        if (isinstance(self.dropout,bool) or not isinstance(self.dropout,Real)
                or not math.isfinite(self.dropout) or not 0<=self.dropout<1):
            raise ValueError("M08: dropout must be finite in [0,1)")


@tf.keras.utils.register_keras_serializable(package="fusion_vsl")
class MaskedTemporalMeanPooling(tf.keras.layers.Layer):
    """[sequence, valid_mask] -> (mean, weights), divide by REAL count.

    Returning uniform real-timestep weights makes ablation debugging explicit.
    All-zero real landmarks are still included; no zero-vector mask inference.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        sequence, valid_mask = inputs
        _temporal_checks(sequence,valid_mask)
        valid = tf.cast(valid_mask,tf.float32)
        weights = valid / tf.reduce_sum(valid,axis=1,keepdims=True)
        clean = tf.where(valid_mask[...,None],sequence,tf.zeros_like(sequence))
        return tf.reduce_sum(clean*weights[...,None],axis=1),weights

    def compute_output_shape(self, input_shape):
        shape=input_shape[0]
        return (shape[0],shape[2]),(shape[0],shape[1])

    def compute_mask(self, inputs, mask=None):
        return None,None

    def get_config(self):
        return super().get_config()


def _branch(features, valid_mask, prefix, dim, options, attention):
    clean=ValidateTemporalInput(feature_dim=dim,name=prefix+"_validate")([features,valid_mask])
    sequence=tf.keras.layers.Dense(options.projection_units,activation="relu",name=prefix+"_projection")(clean)
    sequence=tf.keras.layers.Bidirectional(
        tf.keras.layers.GRU(options.gru_units,return_sequences=True,name=prefix+"_gru"),
        name=prefix+"_bigru")(sequence,mask=valid_mask)
    pool=(MaskedTemporalAttention(options.attention_units,name=prefix+"_pool") if attention else
          MaskedTemporalMeanPooling(name=prefix+"_pool"))
    pooled,_=pool([sequence,valid_mask])
    return pooled


def _fusion_head(rgb_vector, landmark_vector, options, num_classes):
    combined=tf.keras.layers.Concatenate(name="fusion_concat")([rgb_vector,landmark_vector])
    hidden=tf.keras.layers.Dense(options.classifier_units,activation="relu",name="fusion_dense")(combined)
    hidden=tf.keras.layers.Dropout(options.dropout,name="fusion_dropout")(hidden)
    return tf.keras.layers.Dense(num_classes,activation="softmax",dtype="float32",name="probabilities")(hidden)


def _build(rgb_shape, landmark_shape, num_classes, config, attention):
    if tuple(rgb_shape)!=(16,1280) or tuple(landmark_shape)!=(16,300) or type(num_classes) is not int or num_classes!=100:
        raise ValueError("M08: requires RGB(16,1280), Landmark(16,300), 100 classes")
    if config is None:options=FusionModelConfig()
    elif isinstance(config,FusionModelConfig):options=config
    elif isinstance(config,dict):options=FusionModelConfig(**config)
    else:raise ValueError("M08: config must be FusionModelConfig or dict")
    rgb=tf.keras.Input(rgb_shape,dtype="float32",name="rgb_features")
    landmark=tf.keras.Input(landmark_shape,dtype="float32",name="landmark_features")
    valid=tf.keras.Input((16,),dtype="bool",name="valid_mask")
    rgb_vector=_branch(rgb,valid,"rgb",1280,options,attention)
    landmark_vector=_branch(landmark,valid,"landmark",300,options,attention)
    probabilities=_fusion_head(rgb_vector,landmark_vector,options,num_classes)
    name="feature_fusion" if attention else "fusion_without_attention"
    return tf.keras.Model({"rgb_features":rgb,"landmark_features":landmark,"valid_mask":valid},
                          probabilities,name=name)


def build_feature_fusion(rgb_shape=(16,1280), landmark_shape=(16,300), num_classes=100, *, config=None):
    return _build(rgb_shape,landmark_shape,num_classes,config,attention=True)


def build_fusion_without_attention(rgb_shape=(16,1280), landmark_shape=(16,300), num_classes=100, *, config=None):
    return _build(rgb_shape,landmark_shape,num_classes,config,attention=False)


def fusion_inputs(inputs):
    """Select three typed inputs from unchanged M06 dict; never modify arrays.

    M06 detection/padding masks remain trace metadata. Detection absence never
    changes temporal validity. M07 validators check dtype, finite and dimensions.
    """
    rgb=baseline_inputs(inputs,"rgb")
    landmark=baseline_inputs(inputs,"landmark")
    return {"rgb_features":rgb["rgb_features"],"landmark_features":landmark["landmark_features"],
            "valid_mask":rgb["valid_mask"]}


def _pooling_kind(model):
    rgb,landmark=model.get_layer("rgb_pool"),model.get_layer("landmark_pool")
    if isinstance(rgb,MaskedTemporalAttention) and isinstance(landmark,MaskedTemporalAttention):return "attention"
    if isinstance(rgb,MaskedTemporalMeanPooling) and isinstance(landmark,MaskedTemporalMeanPooling):return "masked_mean"
    raise ValueError("M08: unsupported or inconsistent branch pooling")


def fusion_debug_model(model):
    """Share model variables; attention or mean weights, clearly named by kind."""
    kind=_pooling_kind(model)
    suffix="attention_weights" if kind=="attention" else "mean_weights"
    return tf.keras.Model(model.input,{"probabilities":model.output,
            "rgb_"+suffix:model.get_layer("rgb_pool").output[1],
            "landmark_"+suffix:model.get_layer("landmark_pool").output[1]},name=model.name+"_debug")


def _architecture(model):
    if not isinstance(model.input,dict) or set(model.input)!=set(INPUT_CONTRACT):
        raise ValueError("M08: expected a dual-branch Functional model")
    for key,shape in INPUT_CONTRACT.items():
        tensor=model.input[key]
        dtype="bool" if key=="valid_mask" else "float32"
        if tuple(tensor.shape[1:])!=shape or str(tensor.dtype)!=dtype:
            raise ValueError("M08: model tensor contract mismatch")
    if tuple(model.output_shape)!=(None,100):raise ValueError("M08: output must be (B,100)")
    kind=_pooling_kind(model)
    branches={}
    for prefix,dim in (("rgb",1280),("landmark",300)):
        projection=model.get_layer(prefix+"_projection");gru=model.get_layer(prefix+"_bigru")
        if gru.forward_layer.units!=gru.backward_layer.units:
            raise ValueError("M08: asymmetric GRU widths")
        branches[prefix]={"input_dim":dim,"projection_units":projection.units,
                "projection_activation":projection.get_config()["activation"],
                "gru_units_per_direction":gru.forward_layer.units,"return_sequences":True,
                "pooled_dim":2*gru.forward_layer.units,"pooling":kind,
                "attention_units":model.get_layer(prefix+"_pool").attention_units if kind=="attention" else None,
                "explicit_temporal_mask":True}
    rgb_vars={id(v) for layer in ("rgb_projection","rgb_bigru","rgb_pool") for v in model.get_layer(layer).weights}
    landmark_vars={id(v) for layer in ("landmark_projection","landmark_bigru","landmark_pool") for v in model.get_layer(layer).weights}
    if rgb_vars & landmark_vars:raise ValueError("M08: branch variables are unexpectedly shared")
    return {"pooling":kind,"branches":branches,"independent_branch_weights":True,
            "concatenated_dim":int(model.get_layer("fusion_concat").output.shape[-1]),
            "classifier_units":model.get_layer("fusion_dense").units,
            "dropout":model.get_layer("fusion_dropout").rate,"num_classes":100,
            "output_activation":model.get_layer("probabilities").get_config()["activation"]}


def _spec_destination(output_path):
    path=Path(output_path).resolve()
    project=Path(__file__).resolve().parents[2]
    protected=[project/"artifacts/m05/cache",project/"artifacts/manifests",project/"src",project/"scripts",project/"tests"]
    config_path=project/"config.json"
    if config_path.is_file():
        cfg=json.loads(config_path.read_text(encoding="utf-8"))
        protected.extend((project/cfg[name]).resolve() for name in ("handoff_root","data_root"))
    if (path.suffix.lower()!=".json" or path==config_path or any(path.is_relative_to(p) for p in protected)):
        raise ValueError("M08: spec must be a JSON output outside dataset/cache/manifest/source")
    return path


def export_model_spec(model, output_path=None, *, label_mapping, expected_label_mapping_sha256=None,
                      manifest_sha256=None, feature_recipes=None):
    """Export architecture and canonical label binding for an UNTRAINED model.

    This does not load model weights or verify training. Keras save/load alone
    does not carry label semantics: keep this spec alongside the .keras file.
    Optional expected hashes bind to M05/M06 context supplied by the caller.
    """
    if (not isinstance(label_mapping,dict) or len(label_mapping)!=100
            or any(not isinstance(k,str) or not k.strip() or type(v) is not int for k,v in label_mapping.items())
            or sorted(label_mapping.values())!=list(range(100))):
        raise ValueError("M08: label_mapping must preserve 100 names with original IDs 0..99")
    if len({unicodedata.normalize("NFC",k) for k in label_mapping})!=100:
        raise ValueError("M08: Unicode label mapping collision")
    mapping_hash=fingerprint(label_mapping)
    if expected_label_mapping_sha256 is not None and (not is_sha(expected_label_mapping_sha256)
                                                     or mapping_hash!=expected_label_mapping_sha256):
        raise ValueError("M08: canonical label mapping hash mismatch")
    if manifest_sha256 is not None and not is_sha(manifest_sha256):raise ValueError("M08: invalid manifest hash")
    if feature_recipes is not None and (not isinstance(feature_recipes,dict) or set(feature_recipes)!={"rgb","landmark"}
                                        or not all(is_sha(v) for v in feature_recipes.values())):
        raise ValueError("M08: feature_recipes must bind both modality recipe hashes")
    architecture=_architecture(model)
    id_to_name={v:k for k,v in label_mapping.items()}
    spec={"schema_version":VERSION,"model_name":model.name,"trained":False,
          "training_statement":"Architecture/spec only; no training or recognition result certified",
          "architecture":architecture,
          "config":{"projection_units":architecture["branches"]["rgb"]["projection_units"],
                    "gru_units":architecture["branches"]["rgb"]["gru_units_per_direction"],
                    "attention_units":architecture["branches"]["rgb"]["attention_units"],
                    "classifier_units":architecture["classifier_units"],"dropout":architecture["dropout"]},
          "tensor_contracts":{"inputs":{k:{"shape":[None,*shape],"dtype":"bool" if k=="valid_mask" else "float32"}
                                        for k,shape in INPUT_CONTRACT.items()},
                              "output":{"shape":[None,100],"dtype":"float32","kind":"softmax_probabilities"}},
          "mask_policy":{"temporal":"explicit real prefix; padded weights exactly zero; all-invalid error",
                         "missing_landmarks":"zero-filled xyzc retained as real when valid_mask=True",
                         "detection_mask_used_as_model_input":False},
          "label_mapping":dict(label_mapping),"class_names_in_output_order":[id_to_name[i] for i in range(100)],
          "label_mapping_sha256":mapping_hash,"manifest_sha256":manifest_sha256,
          "feature_recipe_sha256":dict(feature_recipes) if feature_recipes else None,
          "model_graph_sha256":fingerprint(model.get_config()),
          "source_sha256":{"fusion_models.py":sha256_file(__file__),
                           "models.py":sha256_file(Path(__file__).with_name("models.py"))}}
    spec["spec_sha256"]=fingerprint(spec)
    if output_path is not None:atomic_write_text(_spec_destination(output_path),canonical(spec)+"\n")
    return spec
