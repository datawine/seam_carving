import cv2
import numpy as np
from debug_module import *
from calc_energy import *

class SeamCarving():
    def __init__(self, img):
        self.img = img

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
        print("find seam")
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
            energy_map = energy(_img)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
#            draw_seam(_img, seam, interactive=False)
            _img = self.delete_single_seam(_img, seam)

        _img = np.rot90(_img).copy()
        print(type(_img))
        for i in range(d_row):
            print("deleting: " + str(i) + "of " + str(d_row) + "lines")
            energy_map = energy(_img)
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
            energy_map = energy(_img)
            dp_map = self.dp(energy_map)
            start_posi = np.argmin(dp_map[-1])
            seam = self.find_seam(dp_map, start_posi)
            seam_list.append(seam)
            _img = self.delete_single_seam(_img, seam)
        
        _img = np.copy(self.img)
        for i in range(len(seam_list)):
            _img = self.add_single_seam(_img, seam_list[i])
            seam_list = self.update_seam_list(seam_list, i)

    def delete_single_seam(self, img, seam):
        r, c, b = img.shape
        output = np.zeros((r, c - 1, b), np.uint8)

        for _c, _r in reversed(seam):
            for _b in range(b):
                output[_r, :, _b] = np.delete(img[_r, :, _b], [_c])

        return output

    def add_single_seam(self, img, seam):
        r, c, b = img.shape
        output = np.zeros((r, c + 1, b), np.uint8)

        for _c, _r in reversed(seam):
            for _b in range(b):
                if _c == 0:
                    insert_seam = np.average(img[_r, _c:_c + 2, b])
                else:
                    insert_seam = np.average(img[_r, _c - 1:_c + 1, b])
                output[_r, :_c, _b] = img[_r, :_c, _b]
                output[_r, _c, _b] = insert_seam
                output[_r, _c + 1:, _b] = img[_r, _c:, _b]
        return output

    def update_seam_list(self, seam_list, index):
        cur_seam = seam_list[index]
        for i in range(index + 1, len(seam_list)):
            for j in range(len(cur_seam)):
                if cur_seam[j] <= seam_list[i][j]:
                    seam_list[i][j] = seam_list[i][j] + 2
        return seam_list