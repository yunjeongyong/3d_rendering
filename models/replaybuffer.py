import torch
import random

class ReplayBuffer:
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.data = []

    # 새로운 이미지를 삽입하고, 이전에 삽입되었던 이미지를 반환하는 함수
    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            # 아직 버퍼가 가득 차지 않았다면, 현재 삽입된 데이터를 반환
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            # 버퍼가 가득 찼다면, 이전에 삽입되었던 이미지를 랜덤하게 반환
            else:
                if random.uniform(0, 1) > 0.5: # 확률은 50%
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element # 버퍼에 들어 있는 이미지 교체
                else:
                    to_return.append(element)
        return torch.cat(to_return)