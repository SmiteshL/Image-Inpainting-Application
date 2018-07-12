#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

cimport numpy as cnp
import cv2
from libc.math cimport sqrt
import numpy as np
from heapq import heappop, heappush



LARGE_VALUE = 1.0e6
KNOWN = 0
BAND = 1
UNKNOWN = 2


cpdef fast_marching_method(cnp.float_t[:, ::1] image, mask, int radius=5):
    

    cdef:
        int i, j,
        cnp.int16_t k, l
        cnp.int16_t[:, ::1] shifted_indices
        cnp.uint8_t[:, ::1] flag
        cnp.float_t[:, ::1] u
        list heap = list()

    flag, u, heap = init_fmm(mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,((radius*2+1),(radius*2+1)))
    indices = np.transpose(np.where(kernel))
    indices_centered = np.ascontiguousarray((indices - (radius)), np.int16)

    while len(heap):
        i, j = heappop(heap)[1]
        flag[i, j] = KNOWN

        if ((i <= 1) or (j <= 1) or (i >= image.shape[0] - 2)
                or (j >= image.shape[1] - 2)):
            continue

        for (k, l) in (i - 1, j), (i, j - 1), (i + 1, j), (i, j + 1):

            if flag[k, l] != KNOWN:
                u[k, l] = min(solve(k - 1, l, k, l - 1, flag, u),
                              solve(k + 1, l, k, l - 1, flag, u),
                              solve(k - 1, l, k, l + 1, flag, u),
                              solve(k + 1, l, k, l + 1, flag, u))

                if flag[k, l] == UNKNOWN:
                    flag[k, l] = BAND
                    heappush(heap, (u[k, l], (k, l)))

                    shifted_indices = (indices_centered + np.array([k, l], np.int16))

                    inpaint_point(k, l, image, flag, u, shifted_indices)


cdef init_fmm(mask):
   
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    outside = cv2.dilate(mask, kernel)
    band = np.logical_xor(mask, outside).astype(np.uint8)

    flag = (2 * outside) - band

    u = np.where(flag == UNKNOWN, LARGE_VALUE, 0)

    heap = []

    indices = np.transpose(np.where(flag == BAND))
    for z in indices:
        heappush(heap, (u[tuple(z)], tuple(z)))

    return flag, u, heap


cdef cnp.float_t solve(int i1, int j1, int i2,
                          int j2, cnp.uint8_t[:, ::1] flag,
                          cnp.float_t[:, ::1] u):
   

    cdef cnp.float_t u_out, u1, u2, perp, s

    u_out = LARGE_VALUE
    u1 = u[i1, j1]
    u2 = u[i2, j2]

    if flag[i1, j1] == KNOWN:
        if flag[i2, j2] == KNOWN:

            perp = sqrt(2 - (u1 - u2) ** 2)

            s = (u1 + u2 - perp) * 0.5
            if s >= u1 and s >= u2:
                u_out = s
            else:
                s += perp
                if s >= u1 and s >= u2:
                    u_out = s
        else:
            u_out = 1 + u1
    elif flag[i2, j2] == KNOWN:
        u_out = 1 + u2

    return u_out


cdef inpaint_point(cnp.int16_t i, cnp.int16_t j, cnp.float_t[:, ::1] image, cnp.uint8_t[:, ::1] flag,
                   cnp.float_t[:, ::1] u,cnp.int16_t[:, ::1] shifted_indices):
   
    cdef:
        cnp.uint8_t[:, ::1] nb
        cnp.int16_t m, n
        cnp.int8_t rx, ry
        cnp.float_t geometric_dst, levelset_dst, direction
        cnp.float_t Ia, Jx, Jy, norm, weight
        cnp.float_t gradx_u, grady_u, gradx_img, grady_img
        int k
    cdef cnp.uint16_t h, w

    h = image.shape[0]
    w = image.shape[1]
    Ia, Jx, Jy, norm = 0, 0, 0, 0

    gradx_u = grad_func(u, i, j, flag)
    grady_u = grad_func(u.T, j, i, flag.T)

    for k in range(shifted_indices.shape[0]):
        m = shifted_indices[k, 0]
        n = shifted_indices[k, 1]

        if m <= 1 or m >= h - 1 or n <= 1 or n >= w - 1:
            continue
        if flag[m, n] != KNOWN:
            continue

        ry = i - m
        rx = j - n

       
        geometric_dst = 1. / ((rx * rx + ry * ry) * sqrt((rx * rx + ry * ry)))
       
        levelset_dst = 1. / (1 + abs(u[m, n] - u[i, j]))
       
        direction = abs(rx * gradx_u + ry * grady_u)

        weight = geometric_dst * levelset_dst * direction

        gradx_img = grad_func(image, m, n, flag)
        grady_img = grad_func(image.T, n, m, flag.T)

        Ia += weight * image[m, n]
        Jx -= weight * gradx_img * rx
        Jy -= weight * grady_img * ry
        norm += weight

   
    image[i, j] = Ia / norm + (Jx + Jy) / sqrt(Jx * Jx + Jy * Jy)


cdef cnp.float_t grad_func(cnp.float_t[:, :] array, int i, int j, cnp.uint8_t[:, :] flag):
  

    cdef:
        cnp.float_t grad

    if flag[i, j + 1] != UNKNOWN:
        if flag[i, j - 1] != UNKNOWN:
            grad = (array[i, j + 1] - array[i, j - 1]) * 0.5
        else:
            grad = (array[i, j + 1] - array[i, j])
    else:
        if flag[i, j - 1] != UNKNOWN:
            grad = (array[i, j] - array[i, j - 1])
        else:
            grad = 0

    return grad
