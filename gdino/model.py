import time
import scipy
import errno
import os
import gdown
import logging
import warnings
from pathlib import Path
from typing import (List, 
                    Tuple, 
                    Union, 
                    Iterable, 
                    Iterator, 
                    TypeVar, 
                    Optional, 
                    Dict
                    )
import gdown

import numpy as np
from numpy import linalg as LA

import torch 
import onnxruntime as ort
from onnxruntime_extensions import get_library_path as _lib_path
from utils.gdino_utils import (generate_masks_with_special_tokens_and_transfer_map, 
                               create_positive_map_from_span
                               )
from gdino.gdino_tokenizer import BertTokenizer

    
T = TypeVar("T")
logging.basicConfig(level=logging.INFO)
ort.set_default_logger_severity(3)


class OnnxGDINO:
    """

    """


    def __init__(
        self, model: str = "gdino_fp32", 
        batch_size: Optional[int] = None, 
        type='gdino_fp32',
        device='cuda',
        trt=False,
        warmup=False,
        n_iters=10
        ):
        """
      
        """ 
        assert device in ['cpu', 'cuda'], 'please use either cuda or cpu!'

        self.providers = [
                    'CPUExecutionProvider'
                ]

        if device == 'cuda':
            self.providers.insert(0, 'CUDAExecutionProvider')

        if trt:
            self.providers.insert(0, ('TensorrtExecutionProvider', {'trt_engine_cache_enable': True, 
                                                                    'trt_max_workspace_size': 4294967296,
                                                                    'trt_engine_cache_path': f'{os.path.dirname(os.path.abspath(__file__))}/data', 
                                                                    'trt_engine_hw_compatible': True,
                                                                    'trt_sparsity_enable': True, 
                                                                    'trt_build_heuristics_enable': True,
                                                                    'trt_builder_optimization_level': 0,
                                                                    'trt_fp16_enable': True
                                                                    }
                        )
            )
     

        if self.providers:
            logging.info(
                "Available providers for ONNXRuntime: ")
            
 

        self.embedding_size = 512

        assert type.split('_')[0] in ['gdino'], 'please choose either: gdino, (quant shortly)'
        self.type = type

        self._model_urls = {'gdino_fp32.onnx': 'https://drive.google.com/file/d/1bdnUeBnMfIhlvswDDMG1L_gSW4URZWt8/view?usp=sharing',}

        vocab_dir = f'{os.path.dirname(os.path.abspath(__file__))}/data/vocab.txt'
        model_dir= f'{os.path.dirname(os.path.abspath(__file__))}/data/{type}.onnx'
                    

        self.tokenizer = BertTokenizer(vocab_file=vocab_dir)
        self.model = self._load_model(model_dir)

        if warmup:
            self.warmup(n_iters=n_iters)

        self._batch_size = batch_size

    @staticmethod
    def sigmoid(x):
        return 1/(1 + np.exp(-x))

    
    def _load_model(self, path: str):

        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        sess_options.graph_optimization_level =  ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess_options.register_custom_ops_library(_lib_path())



        try:
            if os.path.exists(path):
                # `providers` need to be set explicitly since ORT 1.9
                return ort.InferenceSession(
                    path, 
                    sess_options,
                    providers=self.providers,
                
                )
            else:
                raise FileNotFoundError(
                    errno.ENOENT,
                    os.strerror(errno.ENOENT),
                    path,
                )
        except FileNotFoundError:

            gdown.download(url=self._model_urls[os.path.basename(path)], 
                            output=path, 
                            fuzzy=True
                            )
        
            # `providers` need to be set explicitly since ORT 1.9
            return ort.InferenceSession(
                path, 
                sess_options,
                providers=self.providers,
                
            )
      
        
    def warmup(self, n_iters=10):

        payload = self.preprocess_query('time. to. warmup')

        dummy = np.random.randn(1, 3, 800, 1200).astype(np.float32)

        for i in range(n_iters):

            _ , _ = self.model.run(None, {'img': dummy,
                                        'input_ids': np.array(payload['input_ids']),
                                        'attention_mask': np.array(payload['attention_mask']).astype(bool),
                                        'position_ids': payload['position_ids'].detach().numpy(),
                                        'token_type_ids': np.array(payload['token_type_ids']),
                                        'text_token_mask': payload['text_token_mask'].detach().numpy()
                                    }
                                )
        
   
        
    def preprocess_query(self, 
                         query, 
                         max_text_len=256
        ):

        """
        
        separating text preprocessing from image preprocessing
        to allow for single query, multi image inference - or batch preprocessing
        queries

        args:

        query: localisation query for the text encoder
        """

        self.og_query = query
        processed_query = query.lower()
        processed_query = processed_query.strip()
        if not processed_query.endswith("."):
            processed_query = processed_query + "."

        processed_query = [processed_query]
        self.query = processed_query

        inputs = self.tokenizer(processed_query)
        self.inputs = inputs
        special_tokens = self.tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        (
        text_self_attention_masks,
        position_ids,
        cate_to_token_mask_list,
        ) = generate_masks_with_special_tokens_and_transfer_map(
            inputs, special_tokens, self.tokenizer)

        if text_self_attention_masks.shape[1] > max_text_len:
            text_self_attention_masks = text_self_attention_masks[
                :, : max_text_len, : max_text_len]
            
            position_ids = position_ids[:, : max_text_len]
            inputs["input_ids"] = inputs["input_ids"][:, : max_text_len]
            inputs["attention_mask"] = inputs["attention_mask"][:, : max_text_len]
            inputs["token_type_ids"] = inputs["token_type_ids"][:, : max_text_len]

        payload = {}
        payload["input_ids"] = inputs["input_ids"]
        payload["attention_mask"] = inputs["attention_mask"]
        payload["token_type_ids"] = inputs["token_type_ids"]
        payload["position_ids"] = position_ids
        payload["text_token_mask"] = text_self_attention_masks 

        self.inputs = inputs
        
        return payload
    
    def get_phrases_from_posmap(self, 
                                posmap: torch.BoolTensor, 
                                tokenized: Dict, 
                                left_idx: int = 0, 
                                right_idx: int = 255
        ):

        assert isinstance(posmap, torch.Tensor), "posmap must be torch.Tensor"
        if posmap.dim() == 1:
            posmap[0: left_idx + 1] = False
            posmap[right_idx:] = False
            non_zero_idx = posmap.nonzero(as_tuple=True)[0].tolist()
            token_ids = [tokenized["input_ids"][0][i] for i in non_zero_idx]
            return self.tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")
   
       
    def inference(self, 
                  image,
                  payload, 
                  text_threshold=0.25, 
                  box_threshold=0.30,
                  token_spans=None, 
                  with_logits=True):
        

        assert text_threshold is not None or token_spans is not None, "text_threshold and token_spans should not be None at the same time!"
        
        image = np.expand_dims(image, 0)

        logits, boxes = self.model.run(None, {'img': image,
                                    'input_ids': np.array(payload['input_ids']),
                                    'attention_mask': np.array(payload['attention_mask']).astype(bool),
                                    'position_ids': payload['position_ids'].detach().numpy(),
                                    'token_type_ids': np.array(payload['token_type_ids']),
                                    'text_token_mask': payload['text_token_mask'].detach().numpy()
                                }
                            )
        
        
        prediction_logits_ = np.squeeze(logits, 0) 
        prediction_logits_ = self.sigmoid(prediction_logits_)

        prediction_boxes_ = np.squeeze(boxes, 0)
        logits = torch.from_numpy(prediction_logits_)
        boxes = torch.from_numpy(prediction_boxes_) 
    
        # filter output
        if token_spans is None:
            logits_filt = logits.cpu().clone()
            boxes_filt = boxes.cpu().clone()
            filt_mask = logits_filt.max(dim=1)[0] > box_threshold
            logits_filt = logits_filt[filt_mask]  # num_filt, 256
            filtered_boxes = boxes_filt[filt_mask]  # num_filt, 4


            predicted_phrases = []
            for logit, box in zip(logits_filt, boxes_filt):
                pred_phrase = self.get_phrases_from_posmap(logit > text_threshold, self.inputs)
                if with_logits:
                    predicted_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
                else:
                    predicted_phrases.append(pred_phrase)
        else:
            # given-phrase mode
            positive_maps = create_positive_map_from_span(
                self.tokenizer(self.og_query),
                token_span=token_spans
            ).to(image.device) # n_phrase, 256
 
            logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
            all_logits = []
            all_phrases = []
            all_boxes = []
            for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
                # get phrase
                phrase = ' '.join([self.query[_s:_e] for (_s, _e) in token_span])
                filt_mask = logit_phr > box_threshold
                all_boxes.append(boxes[filt_mask])
                all_logits.append(logit_phr[filt_mask])
                if with_logits:
                    logit_phr_num = logit_phr[filt_mask]
                    all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
                else:
                    all_phrases.extend([phrase for _ in range(len(filt_mask))])
            filtered_boxes = torch.cat(all_boxes, dim=0).cpu()
            predicted_phrases = all_phrases

        
        return filtered_boxes, predicted_phrases
        


    



        

