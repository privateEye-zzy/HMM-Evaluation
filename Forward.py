'''
HMM模型：概率计算问题——前向算法，李航《统计学方法》 P177
step1:
    t=0时刻，计算初值：alpha[t=1][j] = pai(j) * B[j][o1]
step2:
    递推：t=1, 2, ..., T-1，计算alpha[t][j]：
    alpha[t][j] = ∑(alpha[t-1][i] * A[i][j]) * B[j][ot]
step3:
    计算：P(O|λ) = ∑alpha[T][j]
'''
import numpy as np
def forward(lamda, O):
    A, B, pai = lamda  # HMM的参数：A, B, pai
    N, M = B.shape  # 状态集合Q的长度N，观测集合V的长度M
    T = len(O)  # 由观测序列得到T个时刻
    '''HMM的前向概率矩阵：T*N'''
    alphas = np.zeros((T, N))
    '''step1：t=0时刻，计算初值：alpha[t=0][j] = pai(j) * B[j][o1]'''
    for j in range(N):  # j是当前状态
        k = V.index(O[0])  # 第Ot个观测对应的V观测集合的位置k:O[0]=V[k]
        alphas[0][j] = pai[j] * B[j][k]
    '''step2：递推：t=1, 2, ..., T-1，计算：alpha[t][j] = ∑(alpha[t-1][i] * A[i][j]) * B[j][Ot]'''
    for t in range(1, T):  # j是当前状态
        k = V.index(O[t])  # 第Ot个观测对应的V观测集合的位置k:O[t]=V[k]
        for j in range(N):
            temp = 0
            for i in range(N):
                temp += alphas[t - 1][i] * A[i][j]  # 计算：∑(alpha[t-1][i] * A[i][j])
            alphas[t][j] = temp * B[j][k]  # 计算：∑(alpha[t-1][i] * A[i][j]) * B[j][Ot]
    '''step3：计算：P(O|λ) = ∑alpha[T][j]'''
    prob = 0  # 求解P(O|λ)
    for j in range(N):  # j是当前隐状态
        prob += alphas[T - 1][j]
    return prob
if __name__ == '__main__':
    '''状态集合：Q = {q1, q2, ..., qN}'''
    Q = ['盒子1', '盒子2', '盒子3']
    '''观测集合：V = {v1, v2, ..., vM}'''
    V = ['红', '白']
    '''
        状态转移概率矩阵：
        A = [aij]，其中aij = P(it+1=qj | it=qi)
        A: shape N * N (盒子之间互相转换)
    '''
    A = np.array([
        [0.5, 0.2, 0.3],
        [0.3, 0.5, 0.2],
        [0.2, 0.3, 0.5]])
    '''
        观测概率矩阵：
        B=[bj(k)]，其中bj(k) = P(ot=vk | it=qj)
        B: shape N * M (N个盒子M个观测)
    '''
    B = np.array([
        [0.5, 0.5],
        [0.4, 0.6],
        [0.7, 0.3]])
    '''初始状态概率向量：pai = P(i1=qi)：第一次出现每个盒子的概率分布'''
    pai = np.array([0.2, 0.4, 0.4])
    '''随机序列1——观测序列：O = {o1, o2, ..., oT}'''
    '''随机序列2——状态序列（隐序列，叠加态的黑盒子）：I = {i1, i2, ..., iT}'''
    O = np.array(['红', '白', '红'])
    '''前向算法计算：P(O|λ)'''
    P_O_or_lamda = forward(lamda=[A, B, pai], O=O)
    print('通过HMM前向算法计算概率：\n观测序列：{} 出现的概率P(O|λ) = {:.4f}'.format(O, P_O_or_lamda))
