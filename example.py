import torch
import torch.nn.functional as F
from diffkd import DiffKD


def main():
    # init DiffKD loss
    diffkd = DiffKD(student_channels=128, teacher_channels=512, 
                    kernel_size=3, use_ae=True, ae_channels=256)
    print(diffkd)
    # get the student feature and teacher feature
    student_feat = torch.randn(2, 128, 7, 7)
    teacher_feat = torch.randn(2, 512, 7, 7)

    student_feat_refined, ddim_loss, teacher_feat_hidden, rec_loss = \
        diffkd(student_feat, teacher_feat)
    kd_loss = F.mse_loss(student_feat_refined, teacher_feat_hidden)
    print(f'kd loss: {kd_loss.item():.4f}    '
          f'ddim loss: {ddim_loss.item():.4f}    '
          f'reconstruction loss: {rec_loss.item():.4f}')

def main_1d():
    # for 1D features such as logits: set kernel_size to 1
    diffkd = DiffKD(student_channels=100, teacher_channels=100, 
                    kernel_size=1, use_ae=False)
    print(diffkd)
    # get the student feature and teacher feature
    student_feat = torch.randn(2, 100)
    teacher_feat = torch.randn(2, 100)
    def _reshape(x):
        return x.view(x.shape[0], x.shape[1], 1, 1)

    student_feat_refined, ddim_loss, teacher_feat, rec_loss = \
        diffkd(_reshape(student_feat), _reshape(teacher_feat))
    kd_loss = F.mse_loss(student_feat_refined, teacher_feat)
    # use KL Div loss for classification predictions
    # kd_loss = F.kl_div(F.log_softmax(student_feat_refined, 1), F.softmax(teacher_feat, 1), reduction="batchmean")
    print(f'kd loss: {kd_loss.item():.4f}    '
          f'ddim loss: {ddim_loss.item():.4f}')


if __name__ == '__main__':
    main()
    # main_1d()
