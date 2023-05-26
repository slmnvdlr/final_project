import math

size = int(60)
a = [0, 0.00625, 0.0125, 0.01875, 0.025, 0.014375]
n = 10 * size
L = 1.0
H = L * 0.1
h = H * 0.025
kp = 1
kp1 = 100
kp2 = 0.0000001
mlz = 0.1
nas = 0
koefA = 1616
delta = L / n
deltaZ = H / size
KMu = 0.16667
centerX = L / 2
wc = 0.95

base_directory = 'D:\\Univer\\Diplom\\Calculations\\e'


d = [0, 0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.7, 0.755, 0.777]
l1 = [0, 0.223, 0.245, 0.3, 0.374, 0.458, 0.503, 0.553, 0.61, 0.668, 0.8, 0.9, 1]


def all_true_len(item_list):
    l = 0
    while l < len(item_list):
        if not item_list[l]:
            item_list.pop(l)
        else:
            l += 1
    return len(item_list)


def kp_test(x, y):
    kptest = False
    if (y - (centerY + h + 0.05)) * (y - (centerY + h + 0.05)) + (x - (centerX + h)) * (x - (centerX + h)) <= (
            R + h) * (R + h) and (y - (centerY - h + 0.05)) * (y - (centerY - h + 0.05)) + (x - (centerX - h)) * (
            x - (centerX - h)) >= (R - h) * (R - h):
        kptest = True
    return kptest


def global_kp_test(y, x, isolation):
    k0_file = open(filenameKD, 'w')
    test_matrix = []
    coordinate = []
    for p in range(y + 1):
        internal = []
        int_coordinate = []
        for k in range(x + 1):
            internal.append(kp_test(k * delta, p * deltaZ))
            int_coordinate.append(k * delta)
        coordinate.append(int_coordinate)
        test_matrix.append(internal)
    for p in range(y):
        for k in range(x):
            item_list = [test_matrix[p][k], test_matrix[p + 1][k + 1], test_matrix[p + 1][k], test_matrix[p][k + 1]]
            if coordinate[p][k] > d[isolation] and coordinate[p][k + 1] < l1[isolation]:
                a_kp = kp2
            else:
                a_kp = kp1
            if not item_list[0]:
                k0_file.write(str(kp) + '\t' + str(k) + '\t' + str(p) + '\n')
            if item_list[0]:
                number = all_true_len(item_list)
                kp3 = (number * a_kp + (len(item_list) - number) * kp) / 4
                k0_file.write(str(kp3) + '\t' + str(k) + '\t' + str(p) + '\n')


def write_file(m):
    open(filenameW, 'w')
    open(filenameF, 'w')
    n = m * 10
    with open(filenameF, 'w') as file:
        file.write(str(n) + '\t' + "//N" + '\n' + str(m) + '\t' + "//	M" + '\n')
    for i in range(0, n + 1):
        Li = delta * i
        with open(filenameF, 'a') as file:
            file.write(str(Li) + '\t' + str(i) + '\n')
        if Li < (1 / 2):
            ctw = koefA * Li + 1
        else:
            ctw = koefA * (1 - Li) + 1
        with open(filenameW, 'a') as file:
            file.write(str(ctw) + '\t' + str(i) + '\n')
    with open(filenameF, 'a') as file:
        for j in range(0, m + 1):
            z = deltaZ * j
            for i in range(0, n + 1):
                file.write(str(z) + '\t' + str(i) + '\t' + str(j) + '\n')


def write_m_lzfun(height):
    open(filenameM, 'w')
    width = height * 10
    with open(filenameM, 'a') as file:
        for j in range(0, height):
            for i in range(0, width):
                file.write(str(mlz) + '\t' + str(i) + '\t' + str(j) + '\n')


#def write_saturation(height):
#    open(filenameS, 'w')
#    width = height * 10
#    with open(filenameS, 'a') as file:
#        for j in range(0, height):
#            for i in range(0, width):
#                file.write(str(nas) + '\t' + str(i) + '\t' + str(j) + '\n')


def write_directory(xi, isolation):
    open(filenameDir, 'w')
    n = '\n'
    base = base_directory + str(xi) + '\\iz' + str(isolation) + '\\'
    grid = base + 'Vhod\\' + 'file.lznet'
    tubes = base + 'Vhod\\' + 'width.lzw'
    porosity = base + 'Vhod\\' + 'm.lzfun'
    permeability = base + 'Vhod\\' + 'kd0.lzfun'
    saturation = 'D:\\Univer\\Diplom\\Calculations\\e'+str(xi)+'\iz0-50\FIELDS\satur_final.lzfun'
    perforation = base + 'Vhod\\' + 'perfor.lzperf'
    ofp = base + 'Vhod\\' + 'relperm.rp'
    settings = base + 'Vhod\\' + 'settings.lzprm'
    with open(filenameDir, 'a') as file:
        file.write(
            base + n + grid + n + tubes + n + porosity + n + permeability + n + saturation + n + perforation + n + ofp + n + settings)


def write_settings(wc):
    set_file = open(filenameSet, 'w')
    n = '\n'
    t = '\t'
    with open(filenameSet, 'a') as file:
        set_file.write('0.1' + t + '0.1' + n + '20' + n + '5' + t + 'true' + t + str(wc) + n + '0.01' + n + '0.0025' + n + '2000' + t + '0')


def write_relperm(K_Mu):
    kmu_file = open(filenameRelp, 'w')
    n = '\n'
    t = '\t'
    with open(filenameRelp, 'a') as flie:
        kmu_file.write(str(K_Mu) + n + '1	1	1	3	3')


def write_perfor():
    perf_file = open(filenamePerf, 'w')
    n = '\n'
    t = '\t'
    with open(filenamePerf, 'a') as flie:
        perf_file.write('1' + n + '0	0.1	0' + n + '1' + n + '0	0.1	0')


def create_all_files(e, iz, grid_size, k_mu, wc):
    write_file(grid_size)
    write_directory(e, iz)
    global_kp_test(grid_size, 10*grid_size, iz)
    write_m_lzfun(grid_size)
    write_perfor()
    write_relperm(k_mu)
#    write_saturation(grid_size)
    write_settings(wc)


for i in range(4, 6):
    if not a[i] == 0:
        R = (4 * a[i] * a[i] + L * L) / (8 * a[i])
    else:
        R = 10000000000
    centerY = - math.sqrt(R * R - centerX * centerX)
    for j in range(4,8):
        mydir = base_directory + str(i) + '\\iz' + str(j) + '\\Vhod\\'
        filenameW = mydir + 'width.lzw'
        filenameKD = mydir + 'kd0.lzfun'
        filenameM = mydir + 'm.lzfun'
    #    filenameS = mydir + 'satur_final.lzfun'
        filenameF = mydir + 'file.lznet'
        filenameDir = mydir + 'kcenter.lzcfg'
        filenameSet = mydir + 'settings.lzprm'
        filenameRelp = mydir + 'relperm.rp'
        filenamePerf = mydir + 'perfor.lzperf'
        create_all_files(i, j, size, KMu, wc)



