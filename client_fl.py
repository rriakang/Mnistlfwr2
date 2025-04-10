import json, logging
import flwr as fl
import time
import os
from collections import OrderedDict
from functools import partial
import warnings
import torch

import client_api
import client_utils

# set log format
handlers_list = [logging.StreamHandler()]

if "MONITORING" in os.environ:
    if os.environ["MONITORING"] == '1':
        handlers_list.append(logging.FileHandler('./fedops/fl_client.log'))
    else:
        pass

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)8.8s] %(message)s",
                    handlers=handlers_list)

logger = logging.getLogger(__name__)

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


class FLClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        validation_split,
        fl_task_id,
        client_mac,
        client_name,
        fl_round,
        gl_model,
        wandb_use,
        wandb_name,
        wandb_run=None,
        model_name=None,
        model_type=None,
        x_train=None,
        y_train=None,
        x_test=None,
        y_test=None,
        train_loader=None,
        val_loader=None,
        test_loader=None,
        cfg=None,
        train_torch=None,
        test_torch=None,
        finetune_llm=None,
        trainset=None,
        tokenizer=None,
        data_collator=None,
        formatting_prompts_func=None,
        num_rounds=None
    ):
        self.cfg = cfg
        self.model_type = model_type
        self.model = model
        self.validation_split = validation_split
        self.fl_task_id = fl_task_id
        self.client_mac = client_mac
        self.client_name = client_name
        self.fl_round = fl_round
        self.gl_model = gl_model
        self.model_name = model_name
        self.wandb_use = wandb_use
        self.wandb_run = wandb_run
        self.wandb_name = wandb_name

        if self.model_type == "Tensorflow":
            self.x_train, self.y_train = x_train, y_train
            self.x_test, self.y_test = x_test, y_test

        elif self.model_type == "Pytorch":
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.train_torch = train_torch
            self.test_torch = test_torch

        elif self.model_type == "Huggingface":
            self.trainset = trainset
            self.tokenizer = tokenizer
            self.finetune_llm = finetune_llm
            self.data_collator = data_collator
            self.formatting_prompts_func = formatting_prompts_func

        elif self.model_type == "hyperparameter":
            # (수정) hyperparameter 모드도 train_loader, val_loader, test_loader를 갖고 있음
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.train_torch = train_torch
            self.test_torch = test_torch

    def set_parameters(self, parameters):
        """Load parameters into local model."""
        if self.model_type in ["Tensorflow"]:
            raise Exception("Not implemented (server-side initialization)")

        elif self.model_type in ["Pytorch"]:
            # Excluding parameters of BN layers
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k]
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=False)

        elif self.model_type in ["Huggingface"]:
            client_utils.set_parameters_for_llm(self.model, parameters)

        elif self.model_type == "hyperparameter":
            # hyperparameter 모드도 pytorch 기반이므로 아래와 같이 처리
            params = [torch.tensor(val) for val in parameters]
            client_utils.set_params(self.model, params)

    def get_parameters(self):
        """Return the current local model parameters."""
        if self.model_type == "Tensorflow":
            raise Exception("Not implemented (server-side parameter initialization)")

        elif self.model_type == "Pytorch":
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "bn" not in name]

        elif self.model_type == "Huggingface":
            return client_utils.get_parameters_for_llm(self.model)

        elif self.model_type == "hyperparameter":
            return [val.cpu().numpy() for val in client_utils.get_params(self.model)]

    def get_properties(self, config):
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """
        Train parameters on the locally held training set.
        This method is called by the server for each training round.
        """
        print(f"config: {config}")

        # 서버에서 내려준 하이퍼파라미터
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]
        lr: float = config["learning_rate"]

        if self.wandb_use:
            self.wandb_run.config.update(
                {"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds},
                allow_val_change=True
            )

        # Round 시작 시간
        round_start_time = time.time()

        # 로컬 모델 저장 경로
        model_path = f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.gl_model}'
        results = {}

        # ─────────────────────────────────────────────────
        # 1. TensorFlow
        # ─────────────────────────────────────────────────
        if self.model_type == "Tensorflow":
            self.model.set_weights(parameters)
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=self.validation_split,
            )
            train_loss = history.history["loss"][-1]
            train_accuracy = history.history["accuracy"][-1]
            val_loss = history.history["val_loss"][-1]
            val_accuracy = history.history["val_accuracy"][-1]

            results = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
            parameters_prime = self.model.get_weights()
            num_examples_train = len(self.x_train)

            self.model.save(model_path + '.h5')

        # ─────────────────────────────────────────────────
        # 2. Pytorch
        # ─────────────────────────────────────────────────
        elif self.model_type == "Pytorch":
            self.set_parameters(parameters)
            trained_model = self.train_torch(self.model, self.train_loader, epochs, self.cfg)

            train_loss, train_accuracy, train_metrics = self.test_torch(trained_model, self.train_loader, self.cfg)
            val_loss, val_accuracy, val_metrics = self.test_torch(trained_model, self.val_loader, self.cfg)

            if train_metrics is not None:
                train_results = {"loss": train_loss, "accuracy": train_accuracy, **train_metrics}
                val_results = {"loss": val_loss, "accuracy": val_accuracy, **val_metrics}
            else:
                train_results = {"loss": train_loss, "accuracy": train_accuracy}
                val_results = {"loss": val_loss, "accuracy": val_accuracy}

            # Prefixing
            train_results_prefixed = {"train_" + k: v for k, v in train_results.items()}
            val_results_prefixed = {"val_" + k: v for k, v in val_results.items()}

            parameters_prime = self.get_parameters()
            num_examples_train = len(self.train_loader)
            torch.save(self.model.state_dict(), model_path + '.pth')

            results = {"fl_task_id": self.fl_task_id, **train_results_prefixed, **val_results_prefixed}

        # ─────────────────────────────────────────────────
        # 3. Huggingface
        # ─────────────────────────────────────────────────
        elif self.model_type == "Huggingface":
            train_results_prefixed = {}
            val_results_prefixed = {}

            self.set_parameters(parameters)
            trained_model = self.finetune_llm(
                self.model, self.trainset, self.tokenizer,
                self.formatting_prompts_func, self.data_collator
            )
            parameters_prime = self.get_parameters()
            num_examples_train = len(self.trainset)

            # 예: train_loss만
            train_loss = results.get("train_loss", None)
            results = {"train_loss": train_loss}

            model_save_path = model_path
            self.model.save_pretrained(model_save_path)
            self.tokenizer.save_pretrained(model_save_path)

        # ─────────────────────────────────────────────────
        # 4. hyperparameter
        # ─────────────────────────────────────────────────
        elif self.model_type == "hyperparameter":
            try:
                # GPU/CPU 설정
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device = device

                logger.info(f"[Client {self.client_name}] train_loader size: {len(self.train_loader.dataset)}")

                # 서버에서 받은 파라미터 적용
                self.set_parameters(parameters)

                # 서버에서 내려준 lr, batch_size, epochs
                # (※ config["learning_rate"], config["batch_size"], config["local_epochs"] 사용)
                # 위에서 이미 batch_size, lr, epochs 추출함

                # 로컬 학습 진행
                loss = client_utils.local_train(
                    model=self.model,
                    train_subset=self.train_loader.dataset,
                    lr=lr,
                    batch_size=batch_size,
                    epochs=epochs,
                    device=self.device
                )

                parameters_prime = self.get_parameters()
                num_examples_train = len(self.train_loader.dataset)

                # 모델 저장
                import os
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(self.model.state_dict(), model_path + '.pth')

                # metrics
                metrics = {
                    "loss": float(loss),
                    "lr": float(lr),
                    "batch_size": int(batch_size)
                }

                # train/val results
                train_results_prefixed = {
                    "train_loss": loss,
                    "train_lr": lr,
                    "train_batch_size": batch_size
                }
                val_results_prefixed = {}

                # 결과 dict
                results = {
                    "fl_task_id": self.fl_task_id,
                    "round": self.fl_round,
                    "gl_model_v": self.gl_model,
                    "train_time": 0.0  # (아래에서 overwrite)
                }

                # train_results_prefixed, val_results_prefixed -> 아래서 합치기

            except Exception as e:
                logger.error(f"[Client {self.client_name}] Exception in fit (hyperparameter): {e}")
                raise e

        else:
            raise ValueError("Unsupported model_type")

        # ─────────────────────────────────────────────────
        # 라운드 종료 시간
        # ─────────────────────────────────────────────────
        round_end_time = time.time() - round_start_time

        # W&B 로깅
        if self.wandb_use:
            self.wandb_run.log({"train_time": round_end_time, "round": self.fl_round}, step=self.fl_round)

            # hyperparameter 모드 등에서 train_results_prefixed/val_results_prefixed가 있으면 여기도 로깅
            # (위에서 변수를 만들었다면 loop를 돌려 푸시)
            if "train_lr" in results:  # 단순 예시
                self.wandb_run.log({"train_lr": results["train_lr"], "round": self.fl_round}, step=self.fl_round)
            # 필요하면 더 추가

        # 학습 후 결과 취합 (train_performance)
        # train_results_prefixed/val_results_prefixed가 없을 수 있으므로 대비
        train_results_prefixed = locals().get("train_results_prefixed", {})
        val_results_prefixed = locals().get("val_results_prefixed", {})

        final_results = {
            "fl_task_id": self.fl_task_id,
            "client_mac": self.client_mac,
            "client_name": self.client_name,
            "round": self.fl_round,
            "gl_model_v": self.gl_model,
            "train_time": round_end_time,
            "wandb_name": self.wandb_name,
            **train_results_prefixed,
            **val_results_prefixed
        }

        json_result = json.dumps(final_results)
        logger.info(f'train_performance - {json_result}')

        # 성능 서버에 결과 전송
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        return (
            locals().get("parameters_prime", []),
            locals().get("num_examples_train", 0),
            {**train_results_prefixed, **val_results_prefixed}
        )

    def evaluate(self, parameters, config):
        """
        Evaluate parameters on the locally held test set.
        This method is called by the server after fit to get validation metrics.
        """
        batch_size: int = config["batch_size"]
        test_loss = 0.0
        test_accuracy = 0.0
        metrics = None
        num_examples_test = 0

        # ─────────────────────────────────────────────────
        # 1. Tensorflow
        # ─────────────────────────────────────────────────
        if self.model_type == "Tensorflow":
            self.model.set_weights(parameters)
            test_loss, test_accuracy = self.model.evaluate(
                x=self.x_test, y=self.y_test, batch_size=batch_size
            )
            num_examples_test = len(self.x_test)

        # ─────────────────────────────────────────────────
        # 2. Pytorch
        # ─────────────────────────────────────────────────
        elif self.model_type == "Pytorch":
            self.set_parameters(parameters)
            test_loss, test_accuracy, metrics = self.test_torch(self.model, self.test_loader, self.cfg)
            num_examples_test = len(self.test_loader)

        # ─────────────────────────────────────────────────
        # 3. Huggingface
        # ─────────────────────────────────────────────────
        elif self.model_type == "Huggingface":
            # 평가 생략
            test_loss = 0.0
            test_accuracy = 0.0
            num_examples_test = 1

        # ─────────────────────────────────────────────────
        # 4. hyperparameter
        # ─────────────────────────────────────────────────
        elif self.model_type == "hyperparameter":
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                self.device = device

                logger.info(f"[Client {self.client_name}] train_loader size: {len(self.train_loader.dataset)}")

                self.set_parameters(parameters)

                lr = config["learning_rate"]
                bs = config["batch_size"]
                local_epochs = config["local_epochs"]

                # 실제로는 evaluate이지만, 현재 구조상
                # hyperparameter 분기에서 "local_train"으로 테스트(?)를 하고 있음
                loss = client_utils.local_train(
                    model=self.model,
                    train_subset=self.train_loader.dataset,
                    lr=lr,
                    batch_size=bs,
                    epochs=local_epochs,
                    device=self.device
                )

                # 업데이트된 파라미터
                parameters_prime = self.get_parameters()
                num_examples_test = len(self.train_loader.dataset)

                metrics = {"loss": float(loss), "lr": float(lr), "batch_size": int(bs)}

                # evaluate에서는 test_loss=loss, test_accuracy=0.0(임의)
                test_loss = float(loss)
                test_accuracy = 0.0

            except Exception as e:
                logger.error(f"[Client {self.client_name}] Exception in evaluate (hyperparameter): {e}")
                raise e

        else:
            raise ValueError("Unsupported model_type")

        # W&B 로깅
        if self.wandb_use:
            self.wandb_run.log({"test_loss": test_loss, "round": self.fl_round}, step=self.fl_round)
            self.wandb_run.log({"test_accuracy": test_accuracy, "round": self.fl_round}, step=self.fl_round)
            if metrics is not None:
                for metric_name, metric_val in metrics.items():
                    self.wandb_run.log({metric_name: metric_val}, step=self.fl_round)

        test_result = {
            "fl_task_id": self.fl_task_id,
            "client_mac": self.client_mac,
            "client_name": self.client_name,
            "round": self.fl_round,
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
            "gl_model_v": self.gl_model,
            "wandb_name": self.wandb_name
        }
        if metrics is not None:
            test_result.update(metrics)

        json_result = json.dumps(test_result)
        logger.info(f'test - {json_result}')

        client_api.ClientServerAPI(self.fl_task_id).put_test_result(json_result)
        self.fl_round += 1

        if metrics is not None:
            # accuracy= test_accuracy, + metrics
            return test_loss, num_examples_test, {"accuracy": test_accuracy, **metrics}
        else:
            return test_loss, num_examples_test, {"accuracy": test_accuracy}


def flower_client_start(FL_server_IP, client):
    """
    Wrapper partial function for starting the flwr client.
    """
    client_start = partial(
        fl.client.start_numpy_client,
        server_address=FL_server_IP,
        client=client
    )
    return client_start
