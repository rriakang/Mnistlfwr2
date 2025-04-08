#client_fl.py

from collections import OrderedDict
import json, logging
import flwr as fl
import time
import os
from functools import partial
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
import warnings
import torch

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

class FLClient(fl.client.NumPyClient):

    def __init__(self, model, validation_split, fl_task_id, client_mac, client_name, fl_round,gl_model, wandb_use, wandb_name,
                 wandb_run=None, model_name=None, model_type=None, x_train=None, y_train=None, x_test=None, y_test=None, 
                 train_loader=None, val_loader=None, test_loader=None, cfg=None, train_torch=None, test_torch=None,
                 finetune_llm=None, trainset=None, tokenizer=None, data_collator=None, formatting_prompts_func=None, num_rounds=None):
        
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
            self.train_loader = train_loader
            self.val_loader = val_loader
            self.test_loader = test_loader
            self.train_torch = train_torch
            self.test_torch = test_torch

    def set_parameters(self, parameters):
        if self.model_type in ["Tensorflow"]:
            raise Exception("Not implemented")
        
        elif self.model_type in ["Pytorch"]:
            keys = [k for k in self.model.state_dict().keys() if "bn" not in k] # Excluding parameters of BN layers
            params_dict = zip(keys, parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # self.model.load_state_dict(state_dict, strict=True)
            self.model.load_state_dict(state_dict, strict=False)

        elif self.model_type in ["Huggingface"]:
            client_utils.set_parameters_for_llm(self.model, parameters)
        
        elif self.model_type == "hyperparameter":
            params = [torch.tensor(val) for val in parameters]
            client_utils.set_params(self.model, params)

            
    
            
    
    def get_parameters(self):
        """Get parameters of the local model."""
        if self.model_type == "Tensorflow":
            raise Exception("Not implemented (server-side parameter initialization)")
        
        elif self.model_type == "Pytorch":
            # Excluding parameters of BN layers
            return [val.cpu().numpy() for name, val in self.model.state_dict().items() if "bn" not in name]
        
        elif self.model_type == "Huggingface":
            return client_utils.get_parameters_for_llm(self.model)
        
        elif self.model_type == "hyperparameter" :
            return [val.cpu().numpy() for val in client_utils.get_params(self.model)]
        

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        print(f"config: {config}")
        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        num_rounds: int = config["num_rounds"]
        # lr : int = config["learning_rate"]
        

        if self.wandb_use:
            # add wandb config
            self.wandb_run.config.update({"batch_size": batch_size, "epochs": epochs, "num_rounds": num_rounds}, allow_val_change=True)

        # start round time
        round_start_time = time.time()

        # model path for saving local model
        model_path = f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.gl_model}'

        # Initialize results
        results = {}
        
        # Training Tensorflow
        if self.model_type == "Tensorflow":
            # Update local model parameters
            self.model.set_weights(parameters)
            
            # Train the model using hyperparameters from config
            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=self.validation_split,
            )

            train_loss = history.history["loss"][len(history.history["loss"])-1]
            train_accuracy = history.history["accuracy"][len(history.history["accuracy"])-1]
            results = {
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": history.history["val_loss"][len(history.history["val_loss"])-1],
                "val_accuracy": history.history["val_accuracy"][len(history.history["val_accuracy"])-1],
            }

            # Return updated model parameters
            parameters_prime = self.model.get_weights()
            num_examples_train = len(self.x_train)

            
            # save local model
            self.model.save(model_path+'.h5')
            

        # Training Torch
        elif self.model_type == "Pytorch":
            # Update local model parameters
            self.set_parameters(parameters)
            
            trained_model = self.train_torch(self.model, self.train_loader, epochs, self.cfg)
            
            train_loss, train_accuracy, train_metrics = self.test_torch(trained_model, self.train_loader, self.cfg)
            val_loss, val_accuracy, val_metrics = self.test_torch(trained_model, self.val_loader, self.cfg)
            
            if train_metrics!=None:
                train_results = {"loss": train_loss,"accuracy": train_accuracy,**train_metrics}
                val_results = {"loss": val_loss,"accuracy": val_accuracy, **val_metrics}
            else:
                train_results = {"loss": train_loss,"accuracy": train_accuracy}
                val_results = {"loss": val_loss,"accuracy": val_accuracy}
                
            # Prefixing keys with 'train_' and 'val_'
            train_results_prefixed = {"train_" + key: value for key, value in train_results.items()}
            val_results_prefixed = {"val_" + key: value for key, value in val_results.items()}

            # Return updated model parameters
            parameters_prime = self.get_parameters()
            num_examples_train = len(self.train_loader)
            
            # Save model weights
            import torch
            torch.save(self.model.state_dict(), model_path+'.pth')

        elif self.model_type == "Huggingface":
            train_results_prefixed = {}
            val_results_prefixed = {}

            # Update local model parameters: LoRA Adapter params
            self.set_parameters(parameters)
            trained_model = self.finetune_llm(self.model, self.trainset, self.tokenizer, self.formatting_prompts_func, self.data_collator)
            parameters_prime = self.get_parameters()
            num_examples_train = len(self.trainset)

            train_loss = results["train_loss"] if "train_loss" in results else None
            results = {"train_loss": train_loss}

            model_save_path = model_path
            self.model.save_pretrained(model_save_path)
            # 선택적으로 tokenizer도 함께 저장
            self.tokenizer.save_pretrained(model_save_path)
            
        elif self.model_type == "hyperparameter":
            train_results_prefixed = {
                    "train_loss": loss,
                    "train_lr": self.lr,
                    "train_batch_size": self.bs
                }
            # hyperparameter에서는 validation set을 사용하지 않으면 빈 dict으로 처리 가능
            val_results_prefixed = {}
            try:
                logger.info(f"[Client {self.client_name}] train_subset size: {len(self.train_subset)}")

                # 1) 모델 파라미터 세팅
                self.set_parameters(parameters)

                # 2) config에서 lr, bs, epochs 추출
                self.lr = config.get("lr", lr)
                self.bs = config.get("batch_size", batch_size)
                epochs = config.get("local_epochs", epochs)

                # 3) 로컬 학습 진행 (local_train은 미리 정의된 함수)
                loss = client_utils.local_train(
                    model=self.model,
                    train_subset=self.train_subset,
                    lr=self.lr,
                    batch_size=self.bs,
                    epochs=epochs,
                    device=self.device
                )

                # 4) 업데이트된 파라미터 리턴
                updated_params = self.get_parameters()
                metrics = {
                    "loss": float(loss),
                    "lr": float(self.lr),
                    "batch_size": int(self.bs)
                }

                num_examples_train = len(self.train_subset)

                # 모델 저장 (선택적)
                torch.save(self.model.state_dict(), f'./local_model/{self.fl_task_id}/{self.model_name}_local_model_V{self.gl_model}.pth')

                results = {
                    "train_loss": loss,
                    "train_lr": self.lr,
                    "train_batch_size": self.bs
                }

            except Exception as e:
                logger.error(f"[Client {self.client_name}] Exception in fit: {e}")
                raise e

        
        else:
            raise ValueError("Unsupported model_type")


        # end round time
        round_end_time = time.time() - round_start_time

        if self.wandb_use:
            # wandb train log
            self.wandb_run.log({"train_time": round_end_time, "round": self.fl_round}, step=self.fl_round)  # train time

            # Log training results
            for key, value in train_results_prefixed.items():
                self.wandb_run.log({key: value, "round": self.fl_round}, step=self.fl_round)

            # Log validation results
            for key, value in val_results_prefixed.items():
                self.wandb_run.log({key: value, "round": self.fl_round}, step=self.fl_round)

        # if train_metrics!=None:
        #     # Training: model performance by round
        #     results = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round, "gl_model_v": self.gl_model,
        #                 "train_time": round_end_time, **train_results_prefixed, **val_results_prefixed}
        # else:
        #     # Training: model performance by round
        #     results = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round, "gl_model_v": self.gl_model,
        #                 "train_time": round_end_time}
        
        results = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round, "gl_model_v": self.gl_model,
                        **train_results_prefixed, **val_results_prefixed,"train_time": round_end_time, 'wandb_name': self.wandb_name}

        json_result = json.dumps(results)
        logger.info(f'train_performance - {json_result}')

        # send train_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_train_result(json_result)

        return parameters_prime, num_examples_train, {**train_results_prefixed, **val_results_prefixed}


    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Get config values
        batch_size: int = config["batch_size"]

        # Initialize test_loss, test_accuracy
        test_loss = 0.0
        test_accuracy = 0.0
        
        metrics=None
        
        if self.model_type == "Tensorflow":
            # Update local model with global parameters
            self.model.set_weights(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy = self.model.evaluate(x=self.x_test, y=self.y_test, batch_size=batch_size)

            num_examples_test = len(self.x_test)
            
        elif self.model_type == "Pytorch":            
            # Update local model parameters
            self.set_parameters(parameters)
            
            # Evaluate global model parameters on the local test data and return results
            test_loss, test_accuracy, metrics = self.test_torch(self.model, self.test_loader, self.cfg)
            num_examples_test = len(self.test_loader)
        
        elif self.model_type == "Huggingface":
            # 평가는 추후 실행
            test_loss = 0.0
            test_accuracy = 0.0
            num_examples_test = 1
        
        elif self.model_type == "hyperparameter":
            self.set_parameters(parameters)
            self.model.eval()

            from torch.utils.data import DataLoader
            import torch.nn as nn

            loader = DataLoader(self.train_subset, batch_size=self.bs, shuffle=False)
            criterion = nn.CrossEntropyLoss()
            total_loss, correct, total = 0.0, 0, 0

            with torch.no_grad():
                for images, labels in loader:
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = self.model(images)
                    batch_loss = criterion(outputs, labels)
                    total_loss += batch_loss.item() * labels.size(0)
                    _, preds = torch.max(outputs, 1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)

            avg_loss = total_loss / total if total > 0 else 0.0
            accuracy = correct / total if total > 0 else 0.0
            num_examples_test = total

            results = {
                "test_loss": avg_loss,
                "test_accuracy": accuracy
            }
        else:
            raise ValueError("Unsupported model_type")

        if self.wandb_use:
            # wandb log
            self.wandb_run.log({"test_loss": test_loss, "round": self.fl_round}, step=self.fl_round)  # loss
            self.wandb_run.log({"test_accuracy": test_accuracy, "round": self.fl_round}, step=self.fl_round)  # acc
            
            if metrics!=None:
                # Log other metrics dynamically
                for metric_name, metric_value in metrics.items():
                    self.wandb_run.log({metric_name: metric_value}, step=self.fl_round)

        # Test: model performance by round
        # if metrics!=None:
        #     test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
        #                 "test_loss": test_loss, "test_accuracy": test_accuracy, **metrics, "gl_model_v": self.gl_model}
        # else:
        #     test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
        #                 "test_loss": test_loss, "test_accuracy": test_accuracy, "gl_model_v": self.gl_model}
        test_result = {"fl_task_id": self.fl_task_id, "client_mac": self.client_mac, "client_name": self.client_name, "round": self.fl_round,
                         "test_loss": test_loss, "test_accuracy": test_accuracy, "gl_model_v": self.gl_model, 'wandb_name': self.wandb_name}
        json_result = json.dumps(test_result)
        logger.info(f'test - {json_result}')

        # send test_result to client_performance pod
        client_api.ClientServerAPI(self.fl_task_id).put_test_result(json_result)

        # increase next round
        self.fl_round += 1

        if metrics!=None:
            return test_loss, num_examples_test, {"accuracy": test_accuracy, **metrics}
        else:
            return test_loss, num_examples_test, {"accuracy": test_accuracy}


def flower_client_start(FL_server_IP, client):
    client_start = partial(fl.client.start_numpy_client, server_address=FL_server_IP, client=client)
    return client_start