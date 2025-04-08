# app.py
import logging
from typing import Dict, Optional, Tuple
import flwr as fl
import datetime
import os
import json
import time
import numpy as np
import shutil
from collections import OrderedDict
from hydra.utils import instantiate
from . import server_api
from . import server_utils
from .server_cluster import GeneticCFLStrategy

# TF warning log filtering
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)


class FLServer():
    def __init__(self, cfg, model, model_name, model_type, gl_val_loader=None, x_val=None, y_val=None, test_torch=None):
        self.task_id = os.environ.get('TASK_ID')  # Set FL Task ID
        self.server = server_utils.FLServerStatus()  # FLServerStatus 인스턴스 생성
        self.model_type = model_type
        self.cfg = cfg
        self.strategy = cfg.server.strategy

        self.batch_size = int(cfg.batch_size)
        self.local_epochs = int(cfg.num_epochs)
        self.num_rounds = int(cfg.num_rounds)
        self.learning_rate = cfg.learning_rate

        self.init_model = model
        self.init_model_name = model_name
        self.next_model = None
        self.next_model_name = None

        if self.model_type == "Tensorflow":
            self.x_val = x_val
            self.y_val = y_val  
        elif self.model_type == "Pytorch":
            self.gl_val_loader = gl_val_loader
            self.test_torch = test_torch
        elif self.model_type == "Huggingface":
            pass
        elif self.model_type == "hyperparameter":
            pass

    def init_gl_model_registration(self, model, gl_model_name) -> None:
        logging.info(f'last_gl_model_v: {self.server.last_gl_model_v}')
        if not model:
            logging.info('init global model making')
            init_model, model_name = self.init_model, self.init_model_name
            print(f'init_gl_model_name: {model_name}')
            self.fl_server_start(init_model, model_name)
            return model_name
        else:
            logging.info('load last global model')
            print(f'last_gl_model_name: {gl_model_name}')
            self.fl_server_start(model, gl_model_name)
            return gl_model_name

    def fl_server_start(self, model, model_name):
        # 모델 파라미터 초기화
        model_parameters = None
        if self.model_type == "Tensorflow":
            model_parameters = model.get_weights()
        elif self.model_type == "Pytorch":
            model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        elif self.model_type == "Huggingface":
            json_path = "./parameter_shapes.json"
            model_parameters = server_utils.load_initial_parameters_from_shape(json_path)
        elif self.model_type == "hyperparameter":
            model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
            logging.info('hyperparameter mode: model_parameters set')

        # 전략(Strategy) 인스턴스화
         # model_parameters 할당 부분...
        if self.model_type == "hyperparameter":
            # 구체적인 on_fit_config_fn을 생성해서 전달합니다.
            on_fit_config_fn = FLServer.get_on_fit_config(
                self.learning_rate, self.batch_size, self.local_epochs, self.num_rounds
            )
            
            strategy_config = {
                "_target_": "server_cluster.GeneticCFLStrategy",  # 실제 클래스의 전체 경로를 사용해야 합니다.
                "init_lr": self.learning_rate,
                "init_bs": self.batch_size,
                "epochs": self.local_epochs,
            }
            
            strategy = instantiate(
                strategy_config,
                initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
                evaluate_fn=self.get_eval_fn(model, model_name),
                on_fit_config_fn=on_fit_config_fn,
                on_evaluate_config_fn=self.evaluate_config
            )



        elif self.model_type in ["Tensorflow", "Pytorch", "Huggingface"]:
            strategy = instantiate(
                self.strategy,
                initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
                evaluate_fn=self.get_eval_fn(model, model_name),
                on_fit_config_fn=self.fit_config,
                on_evaluate_config_fn=self.evaluate_config,
            )
        else:
            raise ValueError("Unsupported model_type")

        # Flower 서버 시작
        fl.server.start_server(
            server_address="0.0.0.0:8080",
            config=fl.server.ServerConfig(num_rounds=self.num_rounds),
            strategy=strategy,
        )

    def get_eval_fn(self, model, model_name):
        """서버 평가 함수 반환: 매 라운드 후 평가 수행"""
        def evaluate(
                server_round: int,
                parameters_ndarrays: fl.common.NDArrays,
                config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            gl_model_path = f'./{model_name}_gl_model_V{self.server.gl_model_v}'
            metrics = None
            if self.model_type == "Tensorflow":
                loss, accuracy = model.evaluate(self.x_val, self.y_val)
                model.set_weights(parameters_ndarrays)
                model.save(gl_model_path + '.tf')
            elif self.model_type == "Pytorch":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                torch.save(model.state_dict(), gl_model_path + '.pth')
            elif self.model_type == "Huggingface":
                logging.warning("Skipping evaluation for Huggingface model")
                loss, accuracy = 0.0, 0.0
                os.makedirs(gl_model_path, exist_ok=True)
                np.savez(os.path.join(gl_model_path, "adapter_parameters.npz"), *parameters_ndarrays)
            elif self.model_type == "hyperparameter":
                import torch
                keys = [k for k in model.state_dict().keys() if "bn" not in k]
                params_dict = zip(keys, parameters_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)
                loss, accuracy, metrics = self.test_torch(model, self.gl_val_loader, self.cfg)
                torch.save(model.state_dict(), gl_model_path + '.pth')
                hyperparam_info = {
                    "lr": config.get("lr", self.learning_rate),
                    "bs": config.get("bs", self.batch_size),
                    "loss": loss,
                    "accuracy": accuracy,
                    "metrics": metrics,
                    "round": self.server.round,
                    "timestamp": time.time()
                }
                logging.info("hyperparam_info : %s", hyperparam_info)
                hyperparam_json_path = gl_model_path + "_hyperparams.json"
                with open(hyperparam_json_path, "w") as f:
                    json.dump(hyperparam_info, f)
            else:
                raise ValueError("Unsupported model_type")

            if self.server.round >= 1:
                self.server.end_by_round = time.time() - self.server.start_by_round
                if metrics is not None:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round,
                                            "gl_loss": loss, "gl_accuracy": accuracy,
                                            "run_time_by_round": self.server.end_by_round,
                                            **metrics, "gl_model_v": self.server.gl_model_v}
                else:
                    server_eval_result = {"fl_task_id": self.task_id, "round": self.server.round,
                                            "gl_loss": loss, "gl_accuracy": accuracy,
                                            "run_time_by_round": self.server.end_by_round,
                                            "gl_model_v": self.server.gl_model_v}
                json_server_eval = json.dumps(server_eval_result)
                logging.info(f'server_eval_result - {json_server_eval}')
                server_api.ServerAPI(self.task_id).put_gl_model_evaluation(json_server_eval)

            if metrics is not None:
                return loss, {"accuracy": accuracy, **metrics}
            else:
                return loss, {"accuracy": accuracy}
        return evaluate

    @staticmethod
    def get_on_fit_config(learning_rate: float, batch_size: int, local_epochs: int, num_rounds: int):
        def fit_config_fn(server_round: int):
            return {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "local_epochs": local_epochs,
                "num_rounds": num_rounds,
            }
        return fit_config_fn

    def fit_config(self, rnd: int):
        fl_config = {
            "batch_size": self.batch_size,
            "local_epochs": self.local_epochs,
            "num_rounds": self.num_rounds,
            "learning_rate": self.learning_rate
        }
        self.server.round += 1
        self.server.start_by_round = time.time()
        logging.info('server start by round')
        return fl_config

    def evaluate_config(self, rnd: int):
        return {"batch_size": self.batch_size}

    def start(self):
        today_time = datetime.datetime.today().strftime('%Y-%m-%d %H-%M-%S')
        self.next_model, self.next_model_name, self.server.last_gl_model_v = \
            server_utils.model_download_s3(self.task_id, self.model_type, self.init_model)
        if self.model_type == "hyperparameter":
            hyperparam_json = f'./{self.next_model_name}_gl_model_V{self.server.last_gl_model_v}_hyperparams.json'
            if os.path.exists(hyperparam_json):
                with open(hyperparam_json, "r") as f:
                    hyperparams = json.load(f)
                logging.info("Loaded hyperparameter info: %s", hyperparams)
        self.server.gl_model_v = self.server.last_gl_model_v + 1
        inform_Payload = {
            "S3_bucket": "fl-gl-model",
            "Last_GL_Model": "gl_model_%s_V.h5" % self.server.last_gl_model_v,
            "FLServer_start": today_time,
            "FLSeReady": True,
            "GL_Model_V": self.server.gl_model_v
        }
        server_status_json = json.dumps(inform_Payload)
        server_api.ServerAPI(self.task_id).put_server_status(server_status_json)
        try:
            fl_start_time = time.time()
            gl_model_name = self.init_gl_model_registration(self.next_model, self.next_model_name)
            fl_end_time = time.time() - fl_start_time
            server_all_time_result = {
                "fl_task_id": self.task_id,
                "server_operation_time": fl_end_time,
                "gl_model_v": self.server.gl_model_v
            }
            json_all_time_result = json.dumps(server_all_time_result)
            logging.info(f'server_operation_time - {json_all_time_result}')
            server_api.ServerAPI(self.task_id).put_server_time_result(json_all_time_result)
            if self.model_type == "Tensorflow":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.h5"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Pytorch":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            elif self.model_type == "Huggingface":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}"
                npz_file_path = f"{global_model_file_name}.npz"
                model_dir = f"{global_model_file_name}"
                real_npz_path = os.path.join(model_dir, "adapter_parameters.npz")
                shutil.copy(real_npz_path, npz_file_path)
                server_utils.upload_model_to_bucket(self.task_id, npz_file_path)
            elif self.model_type == "hyperparameter":
                global_model_file_name = f"{gl_model_name}_gl_model_V{self.server.gl_model_v}.pth"
                server_utils.upload_model_to_bucket(self.task_id, global_model_file_name)
            logging.info(f'upload {global_model_file_name} model in s3')
        except Exception as e:
            logging.error('error: %s', e)
            data_inform = {'FLSeReady': False}
            server_api.ServerAPI(self.task_id).put_server_status(json.dumps(data_inform))
        finally:
            logging.info('server close')
            server_api.ServerAPI(self.task_id).put_fl_round_fin()
            logging.info('global model version upgrade')
