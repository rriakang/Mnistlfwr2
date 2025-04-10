import flwr as fl
from flwr.server.strategy import FedAvg
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import random

def genetic_evolve_2d(vec_list, loss_list, top_k=2, mutation_prob=0.3):
    """
    2차원 벡터 (예: [log10(lr), log2(bs)]) 목록을 GA로 진화.
    
    vec_list: [[lr_log1, bs_log1], [lr_log2, bs_log2], ...]
    loss_list: 해당 클라이언트들의 손실
    top_k: 엘리트로 보존할 상위 개수
    mutation_prob: 돌연변이 발생 확률
    """
    vec_arr = np.array(vec_list)
    loss_arr = np.array(loss_list)

    # 1) loss 오름차순 정렬 후, 상위 top_k 추출
    sorted_indices = np.argsort(loss_arr)
    parents = vec_arr[sorted_indices[:top_k]]  # shape: (top_k, 2)

    # 새 population을 담을 리스트 (초기엔 엘리트 보존)
    new_population = parents.tolist()

    # 2) 나머지 개체 수만큼 parent 중 임의로 2개씩 뽑아 평균
    #    mutation_prob에 따라 돌연변이
    while len(new_population) < len(vec_arr):
        p1, p2 = random.choices(parents, k=2)  # 부모 2명 샘플
        child = (p1 + p2) / 2.0               # 평균으로 crossover
        # 돌연변이
        if random.random() < mutation_prob:
            # 2차원 각각에 대해 약간의 난수 변동
            mutation_vec = (np.random.random(size=2) - 0.5) * 0.5
            child = child + mutation_vec
        new_population.append(child.tolist())

    return new_population

def weighted_average(metrics):
    """
    FedAvg와 같은 방식으로, 클라이언트별 정확도를 가중 평균
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    if sum(examples) == 0:
        return {"accuracy": 0.0}
    return {"accuracy": sum(accuracies) / sum(examples)}

class GeneticCFLStrategy(FedAvg):
    def __init__(self, init_lr=1e-3, init_bs=32, epochs=5, **kwargs):
        super().__init__(
            evaluate_metrics_aggregation_fn=weighted_average,
            **kwargs
        )
        self.cid_to_index = {}
        self.next_index = 0
        self.client_hparams = {}

        # FLServer로부터 받은 초기 설정값 저장
        self.init_lr_log = np.log10(init_lr)
        self.init_bs_log = np.log2(init_bs)
        self.epochs = epochs

    def _get_int_index(self, cid_str: str) -> int:
        if cid_str not in self.cid_to_index:
            self.cid_to_index[cid_str] = self.next_index
            self.next_index += 1
        return self.cid_to_index[cid_str]

    def configure_fit(self, server_round, parameters, client_manager):
        instructions = []
        clients = list(client_manager.all().values())

        for client in clients:
            cid_str = client.cid
            cid_int = self._get_int_index(cid_str)

            # 최초 클라이언트 하이퍼파라미터 설정 시 초기값 사용
            if cid_int not in self.client_hparams:
                self.client_hparams[cid_int] = (self.init_lr_log, self.init_bs_log)

            lr_log, bs_log = self.client_hparams[cid_int]
            lr = 10 ** lr_log
            bs = int(2 ** bs_log)

            fit_config = {
                "lr": float(lr),
                "bs": int(bs),
                "epochs": self.epochs  # YAML에서 받아온 epochs
            }

            fit_ins = fl.common.FitIns(parameters, fit_config)
            instructions.append((client, fit_ins))

        return instructions

    def aggregate_fit(self, server_round, results, failures):
        """
        각 클라이언트의 fit 결과(loss, lr, bs)를 수집한 뒤,
        DBSCAN으로 클러스터링 → 클러스터별 GA 진행 → 클라이언트별 (lr_log, bs_log) 업데이트
        """
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None

        lr_list = []
        bs_list = []
        loss_list = []
        cid_list = []

        for (cid_str, fit_res) in results:
            metrics = fit_res.metrics
            if metrics is None:
                continue

            cid_int = self._get_int_index(cid_str)
            lr_val = metrics["lr"]
            bs_val = metrics["bs"]
            loss_val = metrics["loss"]

            # 여기서 log로 변환하여 저장
            lr_list.append(np.log10(lr_val))
            bs_list.append(np.log2(bs_val))
            loss_list.append(loss_val)
            cid_list.append(cid_int)

        # DBSCAN 군집화 (2차원: [log10(lr), log2(bs)])
        if len(lr_list) > 0:
            X = np.column_stack((lr_list, bs_list))
            X_scaled = StandardScaler().fit_transform(X)

            dbscan = DBSCAN(eps=0.1, min_samples=2)  # eps 등 파라미터 조정 가능
            labels = dbscan.fit_predict(X_scaled)

            num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"[Round {server_round}] DBSCAN clusters: {num_clusters}")

            # 각 클러스터별로 GA 적용
            unique_labels = set(labels)
            for label in unique_labels:
                if label == -1:
                    # outlier는 그대로 두거나, default로 세팅해도 됨
                    continue

                # 해당 클러스터에 속한 클라이언트 인덱스 추출
                cluster_indices = [i for i, lab in enumerate(labels) if lab == label]

                # vec_list = [(lr_log, bs_log)], loss_list
                cluster_vecs = [[lr_list[i], bs_list[i]] for i in cluster_indices]
                cluster_losses = [loss_list[i] for i in cluster_indices]

                # GA로 2D 진화
                new_vecs = genetic_evolve_2d(cluster_vecs, cluster_losses, top_k=2, mutation_prob=0.3)

                # 업데이트된 (lr_log, bs_log) 클라이언트별로 저장
                for idx, new_vec in zip(cluster_indices, new_vecs):
                    c_int = cid_list[idx]
                    self.client_hparams[c_int] = (new_vec[0], new_vec[1])
        else:
            # 클라이언트가 없는 경우
            pass

        return aggregated_parameters

    @staticmethod
    def get_on_fit_config(init_lr: float, init_bs: int, epochs: int, num_rounds: int):
        """
        app.py에서 GeneticCFLStrategy.get_on_fit_config(...)를 호출하려고 할 때 필요한 정적 메서드.
        아래 fit_config_fn을 리턴해주면, Flower 서버에서 on_fit_config_fn=...에 넣어 쓸 수 있습니다.
        """
        def fit_config_fn(server_round: int):
            # 주고자 하는 하이퍼파라미터를 여기서 딕셔너리 형태로 리턴
            # configure_fit와 동일하게 "lr", "bs", "epochs"로 보낼 수도 있고,
            # 필요하다면 "learning_rate", "batch_size" 등으로 맞춰도 됨.
            return {
                "lr": float(init_lr),
                "bs": int(init_bs),
                "epochs": epochs,
                "num_rounds": num_rounds
            }
        return fit_config_fn
ㄴ