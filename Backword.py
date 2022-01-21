'''
HMM模型：概率计算问题——后向算法，李航《统计学方法》 P178
step1:
    t=T时刻，计算初值：beta[t=T][j] = 1
step2:
    递推：t=T-1, T-2, ..., 1，计算beta[t][j]：
    beta[t][i] = ∑(A[i][j] * B[j][ot+1] * beta[t+1][j])
step3:
    计算：P(O|λ) = ∑(PI(j) * B[j][o1] * beta[j][o1])
'''
import numpy as np
def backward(lamda, O):
    A, B, PI = lamda  # HMM的参数：A, B, PI
    N, M = B.shape  # 状态集合Q的长度N，观测集合V的长度M
    T = len(O)  # 由观测序列得到T个时刻
    '''HMM的后向概率矩阵：T*N'''
    betas = np.zeros((T, N))
    '''step1：t=T时刻，计算初值：beta[t=T][j] = 1'''
    for j in range(N):
        betas[-1][j] = 1
    '''step2：递推：t=T-1, T-2, ..., 1，计算：beta[t][i] = ∑(A[i][j] * B[j][ot+1] * beta[t+1][j])'''
    for t in reversed(range(T-1)):
        k = V.index(O[t+1])  # k = [O(t+1) = Vk]
        for i in range(N):
            for j in range(N):
                betas[t][i] += A[i][j] * B[j][k] * betas[t + 1][j]
    '''step3：计算：P(O|λ) = ∑(PI(j) * B[j][o1] * beta[j][o1])'''
    prob = 0  # P(O|λ)
    for j in range(N):
        k = V.index(O[0])  # k = [O(t=0) = Vk]
        prob += PI[j] * B[j][k] * betas[0][j]
    return prob
if __name__ == '__main__':
    '''状态集合：Q = {q1, q2, ..., qN}'''
    Q = ['盒子1', '盒子2', '盒子3']
    '''观测集合：V = {v1, v2, ..., vM}'''
    V = ['红', '白']
    '''# 状态转移概率矩阵：A = [aij]，其中aij = P(it+1 = qj | it = qi)'''
    A = np.array([[0.5, 0.2, 0.3], [0.3, 0.5, 0.2], [0.2, 0.3, 0.5]])  # N*N
    '''观测概率矩阵：B=[bj(k)]，其中bj(k) = P(ot = vk | it = qj)'''
    B = np.array([[0.5, 0.5], [0.4, 0.6], [0.7, 0.3]])  # N*M
    '''初始状态概率向量：pi = P(i1=qi)'''
    PI = np.array([0.2, 0.4, 0.4])
    '''随机序列1——观测序列：0 = {o1, o2, ..., oT}'''
    '''随机序列2——状态序列（隐序列）：I = {i1, i2, ..., iT}'''
    O = np.array(['红', '白', '红'])
    '''后向算法计算：P(O|λ)'''
    P_O_or_lamda = backward(lamda=[A, B, PI], O=O)  # 0.1302
    print('通过HMM后向算法计算概率：\n观测序列：{} 出现的概率P(O|λ) = {:.4f}'.format(O, P_O_or_lamda))
