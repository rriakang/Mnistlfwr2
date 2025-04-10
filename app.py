import logging, json
import socket
import time
from fastapi import FastAPI, BackgroundTasks
import asyncio
import uvicorn
from datetime import datetime

import client_utils
import client_fl
import client_wandb
import client_api


class FLClientTask():
    def __init__(self, cfg, fl_task=None):
        self.app = FastAPI()
        self.status = client_utils.FLClientStatus()
        self.cfg = cfg
        self.client_port = 8003
        self.task_id = cfg.task_id
        self.dataset_name = cfg.dataset.name
        self.output_size = cfg.model.output_size
        self.validation_split = cfg.dataset.validation_split
        self.wandb_use = cfg.wandb.use
        self.model_type = cfg.model_type
        self.model = fl_task["model"]
        self.model_name = fl_task["model_name"]
        
        self.status.client_name = socket.gethostname()
        self.status.task_id = self.task_id
        self.status.client_mac = client_utils.get_mac_address()
        
        logging.info(f'init model_type: {self.model_type}')
        
        if self.wandb_use:
            self.wandb_key = cfg.wandb.key
            self.wandb_account = cfg.wandb.account
            self.wandb_project = cfg.wandb.project
            self.wandb_name = f"{self.status.client_name}-v{self.status.gl_model}({datetime.now()})"

        if self.model_type=="Tensorflow":
            self.x_train = fl_task["x_train"]
            self.x_test = fl_task["x_test"]
            self.y_train = fl_task["y_train"]
            self.y_test = fl_task["y_test"]

        elif self.model_type == "Pytorch":
            self.train_loader = fl_task["train_loader"]
            self.val_loader = fl_task["val_loader"]
            self.test_loader = fl_task["test_loader"]
            self.train_torch = fl_task["train_torch"]
            self.test_torch = fl_task["test_torch"]

        elif self.model_type == "Huggingface":
            self.trainset = fl_task["trainset"]
            self.tokenizer = fl_task["tokenizer"]
            self.finetune_llm = fl_task["finetune_llm"]
            self.data_collator = fl_task["data_collator"]
            self.formatting_prompts_func = fl_task["formatting_prompts_func"]
        
        elif self.model_type == "hyperparameter":
            self.train_loader = fl_task["train_loader"]
            self.val_loader = fl_task["val_loader"]
            self.test_loader = fl_task["test_loader"]
            self.train_torch = fl_task["train_torch"]
            self.test_torch = fl_task["test_torch"]

    async def fl_client_start(self):
        logging.info('FL learning ready')
        logging.info(f'fl_task_id: {self.task_id}')
        logging.info(f'dataset: {self.dataset_name}')
        logging.info(f'output_size: {self.output_size}')
        logging.info(f'validation_split: {self.validation_split}')
        logging.info(f'model_type: {self.model_type}')

        if self.wandb_use:
            logging.info(f'wandb_key: {self.wandb_key}')
            logging.info(f'wandb_account: {self.wandb_account}')
            wandb_run = client_wandb.start_wandb(self.wandb_key, self.wandb_project, self.wandb_name)
        else:
            wandb_run=None
            self.wandb_name=''

        try:
            loop = asyncio.get_event_loop()

            if self.model_type == "Tensorflow":
                client = client_fl.FLClient(
                    model=self.model,
                    x_train=self.x_train,
                    y_train=self.y_train,
                    x_test=self.x_test,
                    y_test=self.y_test,
                    validation_split=self.validation_split,
                    fl_task_id=self.task_id,
                    client_mac=self.status.client_mac,
                    client_name=self.status.client_name,
                    fl_round=1,
                    gl_model=self.status.gl_model,
                    wandb_use=self.wandb_use,
                    wandb_name=self.wandb_name,
                    wandb_run=wandb_run,
                    model_name=self.model_name,
                    model_type=self.model_type
                )
            elif self.model_type == "Pytorch":
                client = client_fl.FLClient(
                    model=self.model,
                    validation_split=self.validation_split,
                    fl_task_id=self.task_id,
                    client_mac=self.status.client_mac,
                    client_name=self.status.client_name,
                    fl_round=1,
                    gl_model=self.status.gl_model,
                    wandb_use=self.wandb_use,
                    wandb_name=self.wandb_name,
                    wandb_run=wandb_run,
                    model_name=self.model_name,
                    model_type=self.model_type,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    test_loader=self.test_loader,
                    cfg=self.cfg,
                    train_torch=self.train_torch,
                    test_torch=self.test_torch
                )
            elif self.model_type == "Huggingface":
                client = client_fl.FLClient(
                    model=self.model,
                    validation_split=self.validation_split,
                    fl_task_id=self.task_id,
                    client_mac=self.status.client_mac,
                    client_name=self.status.client_name,
                    fl_round=1,
                    gl_model=self.status.gl_model,
                    wandb_use=self.wandb_use,
                    wandb_name=self.wandb_name,
                    wandb_run=wandb_run,
                    model_name=self.model_name,
                    model_type=self.model_type,
                    trainset=self.trainset,
                    tokenizer=self.tokenizer,
                    finetune_llm=self.finetune_llm,
                    formatting_prompts_func=self.formatting_prompts_func,
                    data_collator=self.data_collator
                )
            elif self.model_type == "hyperparameter":
                client = client_fl.FLClient(
                    model=self.model,
                    validation_split=self.validation_split,
                    fl_task_id=self.task_id,
                    client_mac=self.status.client_mac,
                    client_name=self.status.client_name,
                    fl_round=1,
                    gl_model=self.status.gl_model,
                    wandb_use=self.wandb_use,
                    wandb_name=self.wandb_name,
                    wandb_run=wandb_run,
                    model_name=self.model_name,
                    model_type=self.model_type,
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    test_loader=self.test_loader,
                    cfg=self.cfg,
                    train_torch=self.train_torch,
                    test_torch=self.test_torch
                )

            client_start = client_fl.flower_client_start(self.status.server_IP, client)

            fl_start_time = time.time()
            await loop.run_in_executor(None, client_start)
            logging.info('fl learning finished')

            fl_end_time = time.time() - fl_start_time

            if self.wandb_use:
                wandb_run.config.update({"dataset": self.dataset_name, "model_architecture": self.model_name}, allow_val_change=True)
                wandb_run.log({"operation_time": fl_end_time, "gl_model_v": self.status.gl_model}, step=self.status.gl_model)
                wandb_run.finish()

                client_wandb.client_system_wandb(
                    self.task_id, self.status.client_mac, self.status.client_name,
                    self.status.gl_model, self.wandb_name, self.wandb_account, self.wandb_project
                )

            client_all_time_result = {
                "fl_task_id": self.task_id,
                "client_mac": self.status.client_mac,
                "client_name": self.status.client_name,
                "operation_time": fl_end_time,
                "gl_model_v": self.status.gl_model
            }
            json_result = json.dumps(client_all_time_result)
            logging.info(f'client_operation_time - {json_result}')

            client_api.ClientServerAPI(self.task_id).put_client_time_result(json_result)

            del client

            self.status.client_start = await client_utils.notify_fin()
            logging.info('FL Client Learning Finish')

        except Exception as e:
            # 문자열 포맷 수정
            logging.info(f"[E][PC0002] learning {e}")
            self.status.client_fail = True
            self.status.client_start = await client_utils.notify_fail()
            raise e

    def start(self):
        @self.app.get('/online')
        async def get_info():
            return self.status

        @self.app.post("/start")
        async def client_start_trigger(background_tasks: BackgroundTasks):
            client_res = client_api.ClientMangerAPI().get_info()
            last_gl_model_v = client_res.json()['GL_Model_V']
            self.status.gl_model = last_gl_model_v

            logging.info('bulid model')
            logging.info('FL start')
            self.status.client_start = True

            self.status.server_IP = client_api.ClientServerAPI(self.task_id).get_port()
            background_tasks.add_task(self.fl_client_start)
            return self.status

        try:
            uvicorn.run(self.app, host='0.0.0.0', port=self.client_port)
        except Exception as e:
            logging.error(f'An error occurred during execution: {e}')
        finally:
            client_api.ClientMangerAPI().get_client_out()
            logging.info(f'{self.status.client_name};{self.status.client_mac}-client close')
