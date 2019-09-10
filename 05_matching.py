import copy
import pyaudio
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.fftpack import fft
import threading

# sheet = [['C4'],['E4', 'G4'], ['G5'], ['C6'], ['D6'], ['E6'], ['A3', 'E4', 'D6'], ['C6'], ['C6'], ['F3','C4'],
#          ['G5'],['C6'],['D6'],['E6'],['G3','D6'],['C6'],['D6'],['B3','D4'],['E6'],['E6'],['C4'],['E4','G4'],['D5'],['C6'],
#          ['D6'],['E6'],['A3','E4','D6'],['C6'],['C6'],['F3','C4'],['G5'],['C6'],['D6'],['E6'], ['G3','D6'],['C6'],['D6'],['B3','D4'],['G6'],['E6']]
# real_note = [['C4'],['E4', 'G4'], ['G5'], ['E6'], ['A3', 'E4', 'E6'], ['C6'], ['C6'], ['F3','C4'],['G5'],['C6'],['D6'],
#              ['E6'],['G3','D6'],['C6'],['D6'],['B3','D4'],['E6'],['E6'],['C4'],['E4','G4'],['D5'],['C6'],['D6'],['E6'],
#              ['A3','E4','D6'],['C6'],['C6'],['F3','C4'],['G5'],['C6'],['D6'],['E6'],['G3','D6'],['C6'],['D6'],['B3','D4'],['G6'],['E6']]

sheet = [['도'],['레'],['미'],['파'],['솔'],['라'],['시'],['도'],['레'],['미']]
real_note = [['도'],['레'],['파'],['솔'],['솔'],['시'],['도'],['레'],['미']]

wait_matching_gyename = []
matching_gyename = []


sheet_match_point = 0
note_match_point = 0
match_matrix = []
matching_result = -1







def matching():
    global sheet
    global sheet_match_point
    global note_match_point
    global matching_gyename
    global wait_matching_gyename
    global match_matrix

    if sheet_match_point == len(sheet) - 3:
        return -1000
    else:
        if len(real_note) > 1:

            if len(wait_matching_gyename) > 0:
                match_matrix.append(sheet[sheet_match_point + 3])  # 악보 상에서 4개 묶기
                wait_matching_gyename.append(matching_gyename)  # 기다린 음이랑 다음 음 2개 묶기

                try:  # 안친거! 안쳐서 틀렸음!
                    a = match_matrix.index(wait_matching_gyename[0])
                    b = match_matrix.index(wait_matching_gyename[1])
                    if (b - a) == 1:  # match_matrix에 그 2개 묶은게 있음
                        if a == 1:
                            sheet_match_point = sheet_match_point + 1
                            matching_gyename = wait_matching_gyename[0]
                            print('1개 안쳤어!! 너무해..ㅠㅠ')
                        elif a == 2:
                            sheet_match_point = sheet_match_point + 2
                            matching_gyename = wait_matching_gyename[0]
                            print('2개 안쳤어!! 너무해..ㅠㅠ')
                        else:
                            print('망함 다시쳐')
                        match_matrix = []
                        wait_matching_gyename = []

                    else:
                        print('음 틀렸음 : ', wait_matching_gyename[0])
                        matching_gyename = wait_matching_gyename[1]
                        wait_matching_gyename = []
                        match_matrix = []
                        sheet_match_point = sheet_match_point + 1
                        note_match_point = note_match_point + 1



                    # 악보에 return하는 표시해야 함!!


                    return (sheet_match_point - 1)  # match_point index를 갖는 곳에(악보에) 틀림 표시
                except ValueError:  # 음을 틀렸음!
                    print('음 틀렸음 : ', wait_matching_gyename[0])
                    matching_gyename = wait_matching_gyename[1]
                    wait_matching_gyename = []
                    match_matrix = []
                    sheet_match_point = sheet_match_point + 1
                    note_match_point = note_match_point + 1

                    return (sheet_match_point - 1)


            else:
                for i in range(0, 3):
                    match_matrix.append(sheet[sheet_match_point + i])

                if match_matrix[0] == matching_gyename:
                    print('matching! gyename :', matching_gyename)
                    sheet_match_point = sheet_match_point + 1
                    note_match_point = note_match_point + 1
                    match_matrix = []
                    matching_gyename = real_note[note_match_point]
                    return -1
                else:
                    print('기다려 : ', matching_gyename)
                    wait_matching_gyename.append(matching_gyename)
                    matching_gyename = real_note[note_match_point + 1]
                    return -1

    # print(match_matrix)

matching_gyename = real_note[0]

for k in range(0, 15):

    # print(matching_gyename)
    matching()
