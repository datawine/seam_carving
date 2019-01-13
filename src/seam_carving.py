import cv2
import numpy as np
from debug_module import *
from calc_energy import *

class SeamCarving():
    def __init__(self, img):
        self.img = img
    
    def get_energy(self, img):
#        return getGrayL1Gradient(img)
#        return getRGBL1Gradient(img)
        return getRGBL2Gradient(img)
#        return getRGBLaplacian(img)
#        return getSaliency(img) + 2 * getGrayL1Gradient(img)

    def get_energy_with_mask(self, img, msk):
        cnt = 0
        raw_energy = self.get_energy(img)
        for i in range(raw_energy.shape[0]):
            for j in range(raw_energy.shape[1]):
                if msk[i][j][0] != 255 and msk[i][j][2] != 255:
                    raw_energy[i][j] = raw_energy[i][j] - 10000000
                    cnt = cnt + 1
        print(cnt)
        return raw_energy

    def dp(self, energy_map):
        print("dping engergy map")
        r, c = energy_map.shape
        dp_map = np.ndarray(shape=(r, c))
        for row in range(r):
            for col in range(c):
                if r == 0:
                    dp_map[row][col] = energy_map[r][c]
                    continue
                if col == 0:
                    dp_map[row][col] = min(dp_map[row - 1][col], dp_map[row - 1][col + 1])
                elif col == c - 1:
                    dp_map[row][col] = min(dp_map[row - 1][col], dp_map[row - 1][col - 1])
                else:
                    dp_map[row][col] = min(dp_map[row - 1][col], dp_map[row - 1][col - 1], dp_map[row - 1][col + 1])                
                dp_map[row][col] = dp_map[row][col] + energy_map[row][col]
        print("dp done")
        return dp_map

    def find_seam(self, dp_map, start_posi, mask=[]):
        r, c = dp_map.shape
        _c = start_posi
        seam = []

        for _r in range(r - 1, -1, -1):
            seam.append([_c, _r])
            right = 2 ** 31
            left = 2 ** 31
            if _r == 0:
                continue
            if _c != 0:
                left = dp_map[_r - 1, _c - 1]
            if _c != c - 1:
                right = dp_map[_r - 1, _c + 1]
            mid = dp_map[_r - 1, _c]
            _c = _c + np.argmin([left, mid, right]) - 1

        print("find seam done")
        return seam

    def seam_delete(self, _row, _col):
        self.col = _col
        self.row = _row
        row, col = self.img.shape[:2]
        d_row = abs(self.row - row)
        d_col = abs(self.col - col)
        print(d_row, d_col)

        _img = np.copy(self.img)

        for i in range(d_col):
            print("deleting: " + str(i) + "of " + str(d_col) + "lines")
            energy_map = self.get_energy(_img)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
#            draw_seam(_img, seam, interactive=False)
            _img = self.delete_single_seam(_img, seam)

        _img = np.rot90(_img).copy()
        print(type(_img))
        for i in range(d_row):
            print("deleting: " + str(i) + "of " + str(d_row) + "lines")
            energy_map = self.get_energy(_img)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
#            draw_seam(_img, seam, interactive=False)
            _img = self.delete_single_seam(_img, seam)

        _img = np.rot90(_img, 3).copy()
        return _img

    def seam_insert(self, _row, _col):
        self.col = _col
        self.row = _row
        row, col = self.img.shape[:2]
        d_row = abs(self.row - row)
        d_col = abs(self.col - col)
        print(d_row, d_col)

        _img = np.copy(self.img)

        seam_list = []
        for i in range(d_col):
            print("deleting: " + str(i) + "of " + str(d_col) + "lines")
            energy_map = self.get_energy(_img)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
            seam_list.append(seam)
            _img = self.delete_single_seam(_img, seam)
        
        _img = np.copy(self.img)
        for i in range(len(seam_list)):
            _img = self.add_single_seam(_img, seam_list[i])
            seam_list = self.update_seam_list(seam_list, i)

        _img_2 = np.rot90(_img).copy()
        seam_list = []
        for i in range(d_row):
            print("deleting: " + str(i) + "of " + str(d_row) + "lines")
            energy_map = self.get_energy(_img_2)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
            seam_list.append(seam)
            _img_2 = self.delete_single_seam(_img_2, seam)
        
        _img = np.rot90(_img).copy()
        for i in range(len(seam_list)):
            _img = self.add_single_seam(_img, seam_list[i])
            seam_list = self.update_seam_list(seam_list, i)

        _img = np.rot90(_img, 3).copy()
        return _img

    def seam_delete_with_mask(self, _row, _col, msk):
        self.col = _col
        self.row = _row
        row, col = self.img.shape[:2]
        d_row = abs(self.row - row)
        d_col = abs(self.col - col)
        print(d_row, d_col)

        _img = np.copy(self.img)
        _msk = np.copy(msk)

        for i in range(d_col):
            print("deleting: " + str(i) + "of " + str(d_col) + "lines")
            energy_map = self.get_energy_with_mask(_img, _msk)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            print("seam val: " + str(dp_map[-1][start_posi]))
            print("start_posi: " + str(start_posi))
            cv2.imwrite("../res/mask_delete" + str(i) + ".jpg", _msk)
            seam = self.find_seam(dp_map, start_posi)
#            draw_seam(_img, seam, interactive=False)
            _img = self.delete_single_seam(_img, seam)
            _msk = self.delete_single_seam(_msk, seam)

        _img = np.rot90(_img).copy()
        _msk = np.rot90(_msk).copy()
        for i in range(d_row):
            print("deleting: " + str(i) + "of " + str(d_row) + "lines")
            energy_map = self.get_energy_with_mask(_img, _msk)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
#            draw_seam(_img, seam, interactive=False)
            _img = self.delete_single_seam(_img, seam)
            _msk = self.delete_single_seam(_msk, seam)

        _img = np.rot90(_img, 3).copy()
        return _img

    def delete_single_seam(self, img, seam):
        r, c, b = img.shape
        output = np.zeros((r, c - 1, b), np.uint8)

        for _c, _r in reversed(seam):
            for _b in range(b):
                output[_r, :, _b] = np.delete(img[_r, :, _b], [_c])

        return output

    def add_single_seam(self, img, seam):
        print("adding seam")
        r, c, b = img.shape
        output = np.zeros((r, c + 1, b), np.uint8)

        for _c, _r in reversed(seam):
            for _b in range(b):
                if _c == 0:
                    insert_seam = np.average(img[_r, _c:_c + 2, _b])
                else:
                    insert_seam = np.average(img[_r, _c - 1:_c + 1, _b])
                output[_r, :_c, _b] = img[_r, :_c, _b]
                output[_r, _c, _b] = insert_seam
                output[_r, _c + 1:, _b] = img[_r, _c:, _b]
        return output

    def update_seam_list(self, seam_list, index):
        cur_seam = seam_list[index]
        for i in range(index + 1, len(seam_list)):
            for j in range(len(cur_seam)):
                if cur_seam[j][0] <= seam_list[i][j][0]:
                    seam_list[i][j][0] = seam_list[i][j][0] + 2
        return seam_list