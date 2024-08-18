# This is a sample Python script.
import torch
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

rotation = torch.eye(3, device=device).unsqueeze(0).expand(pred_keypoints3d.shape[0], -1, -1)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
