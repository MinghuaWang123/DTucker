import os
import argparse
import time
import numpy as np
import cv2
import torch
import h5py
import scipy.io
import tensorly as tl
import torch.nn as nn
from torch.autograd import Variable
from tensorly.decomposition import tucker, parafac, non_negative_tucker
from utils import batch_psnr, normalize, init_logger_ipol, \
    variable_to_cv2_image, remove_dataparallel_wrapper, is_rgb
from scipy.linalg import svd
from functionMLP import SimpleModel, gradient_descent
import torch.optim as optim

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)          # PyTorch随机数种子
    np.random.seed(seed)             # NumPy随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed) # GPU随机数种子
        torch.cuda.manual_seed_all(seed) # 所有GPU随机数种子

def softthre(a, tau):
    return np.sign(a) * np.maximum(np.abs(a) - tau, 0)

def ttm1D(tensor, matrix, mode, sizeC, sizeY_k1):
    # Perform the n-mode product of a tensor and a matrix
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]
    tensor = np.tensordot(tensor, matrix, axes=(mode, 1))
    tensor = np.moveaxis(tensor, -1, mode)
    return tensor


def unfoldx(X, s, i):
    # Create the permutation order
    d = [i] + list(range(i)) + list(range(i + 1, X.ndim))
    # Permute the array dimensions
    X = np.transpose(X, axes=d)
    # Reshape the array
    X = np.reshape(X, (s[i], -1))

    return X


def compute_psnr(x_true, x_pred):
    assert x_true.ndim == 3 and x_pred.ndim == 3
    img_w, img_h, img_c = x_true.shape
    ref = x_true.reshape(-1, img_c)
    tar = x_pred.reshape(-1, img_c)
    msr = np.mean((ref - tar) ** 2, 0)
    max2 = np.max(ref, 0) ** 2
    psnrall = 10 * np.log10(max2 / msr)
    m_psnr = np.mean(psnrall)
    psnr_all = psnrall.reshape(img_c)
    return m_psnr


def calculate_mean_psnr_per_band(img1, img2):
    # 确保图像具有相同的形状
    assert img1.shape == img2.shape, "The images must have the same dimensions."

    bands = img1.shape[-1]  # 获取波段数
    psnr_values = []

    for band in range(bands):
        psnr = compute_psnr(img1[..., band], img2[..., band])
        psnr_values.append(psnr)

    mean_psnr = np.mean(psnr_values)
    return mean_psnr

# 定义损失函数为两个张量的F范数（欧式距离）
def frobenius_norm_loss(X, Y):
    # 计算X与Y之间的差值
    diff = X - Y
    # 计算差值的F范数
    loss = torch.norm(diff, p='fro')
    return loss


def main_finally(**args):
    # Init logger
    logger = init_logger_ipol()
    cuda_ = args['cuda']
    # add_noise = args['add_noise']
    # noise_sigma = args['noise_sigma']
    im_path = args['input']
    if im_path.endswith('.mat'):
        img = scipy.io.loadmat(im_path)
        oriData3_noise = img['oriData3_noise']
        gt = img['data_HSI']
        X = img['data_MSI']
    else:
        print("Please input .mat file!")
    # PAV02




    # PAV01
    # rank1 = [150, 150, 3]
    # rank2 = [150, 150, 3]
    # beta = 0.15
    # lambdaa = 0.1  # 0.1 33.13
    # Iter = 10
    # lambda1 = 0.05
    # lambda2 = 0.15
    # mu = 0.001
    # alpha1 = 0.15
    # alpha2 = 0.005

    # PAV04
    rank1 = [150, 150, 3]
    rank2 = [150, 150, 3]
    beta = 0.15
    lambdaa = 0.1  # 0.1 33.13
    Iter = 30
    lambda1 = 0.05
    lambda2 = 0.15
    mu = 0.001
    alpha1 = 0.15
    alpha2 = 0.001
    set_seed(42)


    [C1, U1] = tucker(oriData3_noise, rank1)
    Q1 = 0  # np.zeros(C1.shape)
    Lam1 = Q1
    [C2, U2] = tucker(X, rank2)
    Q2 = 0  # np.zeros(C2.shape)
    Lam2 = Q2
    [c1_m, c1_n, c1_z] = C1.shape
    [c2_m, c2_n, c2_z] = C2.shape
    C1_matrix = np.reshape(C1, (c1_m * c1_n, c1_z))
    C2_matrix = np.reshape(C2, (c2_m * c2_n, c2_z))
    Y = oriData3_noise.copy()
    sizeY = list(Y.shape)
    sizeX = list(X.shape)
    S = np.zeros(Y.shape)
    errList = []
    normD = np.linalg.norm(Y.flatten())

    for i in range(Iter):
        # update U1
        for j in range (Y.ndim):
            unfoTemp = unfoldx(oriData3_noise - S, sizeY, j)
            tempC = C1.copy()
            sizeC = rank1.copy()
            for k1 in [k for k in range(Y.ndim) if k != j]:
                tempC = ttm1D(tempC, U1[k1], k1, rank1, sizeY[k1])
                sizeC[k1] = sizeY[k1]
            UnfoldC = unfoldx(tempC, rank1, j)
            tempMatix = np.dot(unfoTemp, UnfoldC.T)
            V1, _, V2 = svd(tempMatix, full_matrices=False)
            U1[j] = np.dot(V1, V2)

            #update Q1
            Q1 = softthre(C1 + Lam1 / mu, lambda1 / mu)
            Q1_matrix = np.reshape(Q1, (c1_m * c1_n, c1_z))

            # networks for Phi
            # 设置模型参数
            input_size = 3
            hidden_size = 10 # 隐藏层大小可以自由设置
            output_size = 3
            negative_slope = 0.01
            model = SimpleModel(input_size, hidden_size, output_size, negative_slope)

            # 定义损失函数和优化器
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            # optimizer = optim.Adam(model.parameters(), lr=0.01, betas=(0.9, 0.999))  # , weight_decay=0.00001

            # 训练模型
            for epoch in range(10):
                # 前向传播
                inputC1 = torch.tensor(C1_matrix, dtype=torch.float32)
                output = model(inputC1)
                loss = criterion(output, torch.tensor(C2_matrix, dtype=torch.float32))

                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if epoch % 10 == 0:
                    # print(f'Epoch [{epoch}/100], Loss: {loss.item():.4f}')

            # 获取模型参数
            W1 = model.linear1.weight.detach().numpy()
            b1 = model.linear1.bias.detach().numpy()

            W2 = model.linear2.weight.detach().numpy()
            b2 = model.linear2.bias.detach().numpy()


            # update C1
            dC11 = gradient_descent(C1_matrix.T, C2_matrix.T, W1, b1, W2, b2, beta, negative_slope=0.01)
            sizeD = rank1.copy()
            Y1 = C1.copy()
            for ky in range(Y.ndim):
                Y1 = ttm1D(Y1, U1[ky], ky, sizeD, sizeY[ky])
                sizeD[ky] = sizeY[ky]
            dC13 = mu * (C1 - Q1 + Lam1/mu)
            dC12 = oriData3_noise - Y1 - S
            sizeCY = sizeY.copy()
            for k in range(Y.ndim):
                dC12 = ttm1D(dC12, U1[k].T, k, sizeCY, rank1[k])
                sizeCY[k] = rank1[k]

            C1_matrix -= alpha1 * (dC11.T + np.reshape(dC13, (c1_m*c1_n, c1_z)) - np.reshape(dC12, (c1_m*c1_n, c1_z)))
            C1 = np.reshape(C1_matrix, (c1_m, c1_n, c1_z))

            # update  Y
            preY = Y.copy()
            sizeD = rank1.copy()
            Y1 = C1.copy()
            for ky in range(Y.ndim):
                Y1 = ttm1D(Y1, U1[ky], ky, sizeD, sizeY[ky])
                sizeD[ky] = sizeY[ky]
            Y = Y1.copy()

            # update U2
            for j2 in range(Y.ndim):
                unfoTemp2 = unfoldx(X, sizeX, j2)
                sizeC2 = rank2.copy()
                tempu2 = C2.copy()
                for k2 in [kj for kj in range(Y.ndim) if kj != j2]:
                    tempu2 = ttm1D(tempu2, U2[k2], k2, sizeC2, sizeX[k2])
                    sizeC2[k2] = sizeX[k2]
                UnfoldC2 = unfoldx(tempu2, rank2, j2)
                tempMatix2 = np.dot(unfoTemp2, UnfoldC2.T)
                V12, _, V22 = svd(tempMatix2, full_matrices=False)
                U2[j2] = np.dot(V12, V22.T)
            # update Q2
            Q2 = softthre(C2 + Lam2 / mu, lambda2 / mu)
            Q2_matrix = np.reshape(Q2, (c2_m * c2_n, c2_z))

            # update C2
            TempC21 = beta * (np.reshape(output.detach().numpy(),(c2_m, c2_n, c2_z)) -C2)
            TempC23 = mu * (C2 - Q2 + Lam2/mu)
            TempC22 = X.copy()
            sizeCX = sizeX.copy()
            for kc2 in range(Y.ndim):
                TempC22 = ttm1D(TempC22, U2[kc2].T,kc2,sizeCX,rank2[kc2])
                sizeCX[kc2] = rank2[kc2]
            C2 -= alpha2 * (TempC23 - TempC21 - TempC22 )
            C2_matrix = np.reshape(C2, (c2_m * c2_n, c2_z))

            # update S
            S = softthre(oriData3_noise - Y, lambdaa)


            # update Lam12
            Lam1 = Lam1 + mu * (C1 - Q1)
            Lam2 = Lam2 + mu * (C2 - Q2)
            mu = mu * 1.5
            epsilon = 0.00001

            err = np.linalg.norm(Y.flatten() - preY.flatten()) / normD
            errList.append(err)
            # Check for convergence
            if err < epsilon:
                break


        # Print the iteration and the error
        print(f'JD: iterations = {i + 1}  lambdaa:{lambdaa}  difference={err:.6f}')
        mpsnr = compute_psnr(gt, Y)
        print('psnr:', format(mpsnr, '.2f'))


        scipy.io.savemat('recovereddata_GF.mat', {'output': Y})
        # mpsnr = compute_psnr(gt, Y)
        # print('psnr:', format(mpsnr, '.2f'))


if __name__ == "__main__":
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # Parse arguments
    parser = argparse.ArgumentParser(description="TuckerJoDe")
    parser.add_argument("--input", type=str, default="D:\\PAV04.mat", help='path to input image')
    # parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    parser.add_argument("--no_gpu", action='store_true', help="run model on CPU")
    argspar = parser.parse_args()

    # use CUDA?
    argspar.cuda = not argspar.no_gpu and torch.cuda.is_available()

    print("\n### Testing JDMLP model ###")
    print("> Parameters:")
    for p, v in zip(argspar.__dict__.keys(), argspar.__dict__.values()):
        print('\t{}: {}'.format(p, v))
    print('\n')

    main_finally(**vars(argspar))