import numpy as np
import pandas as pd
import copy
import os
import time
from shutil import copyfile
import subprocess

import matplotlib.pyplot as plt

from multiprocessing import Pool
import multiprocessing
import threading

from scipy.signal import find_peaks
from scipy import interpolate

from ptb.util.gait.helpers import OsimHelper
from ptb.util.gait.opsim import IK as IKT
from ptb.util.gait.defaults import OsimIKLabels
from ptb.util.data import StorageIO, StorageType, Yatsdo, TRC, MocapDO, MYXML
from ptb.util.math.filters import Butterworth

from opensim import InverseKinematicsTool


class Analysis:

    @staticmethod
    def worker(job):
        data = job[1]
        model_file = job[0]
        model = OsimHelper(model_file)
        model.set_joints(data)
        return model.markerset

    @staticmethod
    def worker2(job):
        data = job[1]
        model_file = job[0]
        model = OsimHelper(model_file)
        ret = []
        for i in range(0, data.shape[0]):
            model.set_joints(data.iloc[i, :])
            ret.append(copy.deepcopy(model.markerset))
        return ret

    @staticmethod
    def debug_print_swing(keyw, x, ar, kd):
        plt.figure()
        plt.title("swings: " + keyw)
        plt.plot(x[ar], kd[ar], 'x')
        plt.plot(x, kd)
        plt.show()

    @staticmethod
    def debug_print_stance(keyw, x, ar, kd):
        plt.figure()
        plt.title("stance: " + keyw)
        plt.plot(x[ar], kd[ar], 'x')
        plt.plot(x, kd)
        plt.show()
        pass

    @staticmethod
    def write_trc(marker_data, filename="./test.trc"):
        time.sleep(1)
        marker_data.write(filename)
        print("Marker data exported")

    @staticmethod
    def get_A_in_B_str(target, the_list):
        for t in the_list:
            if target.lower() == t.lower():
                return t
        return None

    @staticmethod
    def find_stride(ikfile, model_file, is_debug=True, percentage_cpu=0.25, target_markers=None, frame_number_st=0, show_final=False):
        """
        This Code uses the knee flexion with the associated osim model to estimate the strides
        Use this when you don't have acceleration or force data
        :param ikfile: STO/MOT file of the joint angles (Opensim Output)
        :param model_file:
        :param is_debug:
        :return: A list of strides (data chopped into strides)
        """
        ikfilename = os.path.split(ikfile)
        modelfilename = os.path.split(model_file)
        sto = None
        try:
            sto = StorageIO.load(ikfile, StorageType.mot)# load IK file
        except pd.errors.EmptyDataError:
            return None
        data_col = [c for c in sto.data.columns]
        npf = np.zeros(sto.data.shape)
        npf[:, 0] = sto.data.iloc[:, 0]
        for i in range(1, sto.data.shape[1]):
            y = sto.data.iloc[:, i]
            yf = Butterworth.butter_low_filter(y, 4, 100, 4)
            npf[:, i] = yf
            pass
        sto.data = pd.DataFrame(data=npf, columns=data_col)

        if target_markers is None:
            # target_markers = {'right': {'heel': 'RHeel', 'toe': 'RToe'}, 'left': {'heel': 'LHeel', 'toe': 'LToe'}}
            target_markers = {'right': {'heel': 'RHeel', 'toe': 'RMT2'}, 'left': {'heel': 'LHeel', 'toe': 'LMT2'}}
        # Generate marker data from model
        marker_data = None
        model_name = None
        if model_file is not None:
            start = time.time()
            parts = os.path.split(model_file)
            if not os.path.exists("./temp/"):
                os.makedirs("./temp/")
            # create a temp folder
            model_temp = "./temp/{0}".format(parts[1])
            copyfile(model_file, model_temp)
            files = os.listdir("./temp/")
            temp_model = Analysis.get_A_in_B_str(parts[1], files)
            if temp_model is None:
                # return if not created
                return
            model_temp = "./temp/{0}".format(temp_model)
            modelfilename_checked = "{0}/{1}".format(parts[0], temp_model)
            model = OsimHelper(modelfilename_checked) # load model
            model_name = model.osim_model.osim.getName()
            # create trc template
            markers_l = [m for m in model.markerset]
            trc_cols = ["Frame#", "Time"]
            for m in range(0, len(markers_l)):
                trc_cols.append("{0}_X{1}".format(markers_l[m], m + 1))
                trc_cols.append("{0}_Y{1}".format(markers_l[m], m + 1))
                trc_cols.append("{0}_Z{1}".format(markers_l[m], m + 1))


            # Set up data blocks to be processed
            bx = [b for b in range(0, sto.data.shape[0], int(1/sto.dt))]
            bx.append(sto.data.shape[0])

            # Create work list for multiprocessing pool
            work = [[model_temp, sto.data.iloc[bx[x]: bx[x+1], 1:], sto.data.shape[0]] for x in range(0, len(bx)-1)]
            ret = []
            if percentage_cpu < 0:
                ret = [Analysis.worker2(w) for w in work]
            else:
                try:
                    # define the number of workers
                    num_workers = int(multiprocessing.cpu_count()*percentage_cpu)

                    # create the pool of workers
                    p = Pool(num_workers)
                    # run jobs
                    ret = p.map(Analysis.worker2, work)
                except Exception:
                    ret = [Analysis.worker2(w) for w in work]
                    pass

            def reshape(dp):
                n = dp.to_numpy()
                return n.reshape([1, n.shape[0]*n.shape[1]], order='F')

            def reshape_block(dp):
                nx = np.zeros([len(dp), len(markers_l)*3])
                for i in range(0, len(dp)):
                    nx[i, :] = reshape(dp[i])
                return nx

            # reformating results
            markers = np.zeros([sto.data.shape[0], len(markers_l)*3+2])
            markers[:, 1] = sto.data.iloc[:, 0]
            for frame in range(0, len(ret)):
                g = reshape_block(ret[frame])
                markers[bx[frame]:bx[frame + 1], 0] = [int(bx[frame] + i + 1) for i in range(0, g.shape[0])]
                markers[bx[frame]:bx[frame + 1], 2:] = g

            # Create trc object
            trc_pd = pd.DataFrame(data=markers, columns=trc_cols)
            marker_data = TRC(trc_pd)
            marker_data.update()

            end = time.time()

            # write out using thread so it doesn't block
            mx = threading.Thread(target=Analysis.write_trc, args=(marker_data, "./test.trc",))
            mx.start()

            print("Generated marker data from model and IK data in {0:.3f}s".format(end-start))

        # joint angle to left right label mapping
        leg = {"knee_angle_r": "right",
               "knee_angle_l": "left"
               }
        x = np.array(sto.data.iloc[:, 0].to_list())
        dt = np.mean(x[1:]-x[:-1])

        swings = {"knee_angle_r": [], "knee_angle_l": []}
        for keyw in ["knee_angle_r", "knee_angle_l"]:
        # for keyw in ["ankle_angle_r", "ankle_angle_l"]:
            kd = np.array(sto.data[keyw].to_list())
            kdb = Butterworth.butter_low_filter(kd, 8, 1/dt)
            m = np.max(kdb)
            n = np.min(kdb)
            if np.abs(n) > np.abs(m):
                kdb = -kdb
                m = np.max(kdb)
                n = np.min(kdb)
            mn = (m-n)/2
            ar, _ = find_peaks(kdb, distance=20, height=0.5*mn + n)
            distance = np.mean([ar[d] - ar[d - 1] for d in range(1, len(ar))]) * 0.65
            ar, _ = find_peaks(kdb, distance=distance, height=0.5*mn + n)

            stq = kdb[0]/np.mean(kdb[ar])
            stq1 = kdb[-1] / np.mean(kdb[ar])
            if stq > 0.80:
                stqr = [0]
                for i in range(0, ar.shape[0]):
                    stqr.append(ar[i])
                ar = np.array(stqr)

            if stq1 > 0.80:
                stqr = []
                for i in range(0, ar.shape[0]):
                    stqr.append(ar[i])
                stqr.append(kdb.shape[0]-1)
                ar = np.array(stqr)

            mid = np.median([kdb[ar[i]] for i in range(0, ar.shape[0])])
            percent_mid = [kdb[ar[i]]/mid for i in range(0, ar.shape[0])]
            filter_ar = [ar[i] for i in range(0, ar.shape[0]) if percent_mid[i] > 0.8]
            old_ar = copy.deepcopy(ar)
            ar = filter_ar
            # swings[keyw] = ar
            if is_debug:
                try:
                    Analysis.debug_print_swing(keyw, x, ar, kd)
                except IndexError:
                    ar = old_ar
                    Analysis.debug_print_swing(keyw, x, ar, kd)
            swings[keyw] = ar

        stance = {"knee_angle_r": [], "knee_angle_l": []}
        for keyw in ["knee_angle_r", "knee_angle_l"]:
            kd = np.array(sto.data[keyw].to_list())
            kdb = Butterworth.butter_low_filter(kd, 8, 1/dt)
            m = np.max(kdb)
            n = np.min(kdb)
            if np.abs(n) < np.abs(m):
                kdb = -kdb
                m = np.max(kdb)
                n = np.min(kdb)
            mn = (m-n)/2
            # ar, _ = find_peaks(kdb, distance=20, height=mn + n)
            ar, _ = find_peaks(kdb, distance=20, height=(mn + n)*0.5)
            distance = np.mean([ar[d]-ar[d-1] for d in range(1, len(ar))])*0.75
            # ar, _ = find_peaks(kdb, distance=distance, height=mn + n)
            ar, _ = find_peaks(kdb, distance=distance, height=(mn + n)*0.5)
            stance[keyw] = ar
            if is_debug:
                Analysis.debug_print_stance(keyw, x, ar, kd)
            pass

        heel = {"knee_angle_r": [], "knee_angle_l": []}
        toe = {"knee_angle_r": [], "knee_angle_l": []}

        for keyw in ["knee_angle_r", "knee_angle_l"]:
            for s in swings[keyw]:
                diffs = np.array(stance[keyw]) - s
                k0 = np.where(diffs > 0)
                try:
                    heel[keyw].append(stance[keyw][k0[0][0]])
                except KeyError:
                    heel[keyw].append(np.nan)
                    pass
                except IndexError:
                    heel[keyw].append(np.nan)
                    pass
                k1 = np.where(diffs < 0)
                try:
                    toe[keyw].append(stance[keyw][k1[0][-1]])
                except KeyError:
                    toe[keyw].append(np.nan)
                    pass
                except IndexError:
                    toe[keyw].append(np.nan)
                    pass
            pass

        heel_marker = {"knee_angle_r": marker_data.marker_set[target_markers['right']['heel']],
                       "knee_angle_l": marker_data.marker_set[target_markers['left']['heel']]}
        toe_marker = {"knee_angle_r": marker_data.marker_set[target_markers['right']['toe']],
                      "knee_angle_l": marker_data.marker_set[target_markers['left']['toe']]}
        heel_strike = {"knee_angle_r": [], "knee_angle_l": []}
        toe_off = {"knee_angle_r": [], "knee_angle_l": []}
        top_label = [m for m in marker_data.marker_names if "sternum" in m.lower()]
        ptop1 =marker_data.marker_set[top_label[0]].to_numpy()
        pelvis_label = [m for m in marker_data.marker_names if "rpsis" in m.lower() or "r_psis" in m.lower() or 'pelvis4' in m.lower()]
        # try:
        #     a = pelvis_label[0]
        # except IndexError:
        #     print("pelvis error line 397")
        #     return None
        ptop0 = marker_data.marker_set[pelvis_label[0]].to_numpy()
        ptop2 = np.abs(ptop1-ptop0)
        masx = np.max(ptop2, axis=0)
        up = 0
        for i in range(1, masx.shape[0]):
            if masx[up] < masx[i]:
                up = i
        for keyw in ["knee_angle_r", "knee_angle_l"]:
            for h in toe[keyw]:
                tagger = frame_number_st+h
                st = 80
                if np.isnan(h):
                    continue
                if st > h:
                    st = h
                print(keyw)
                print(tagger-st)
                print(tagger)
                print(tagger+80)
                ys = toe_marker[keyw].iloc[h-st:h+80, :].to_numpy()
                y = [np.linalg.norm(ys[yx, up]) for yx in range(0, ys.shape[0])]
                y0 = [c/np.max(np.array(y)) for c in y]
                kdb = Butterworth.butter_low_filter(y0, 6, 1 / dt)
                x = np.array([i for i in range(0, len(y))])
                p = interpolate.InterpolatedUnivariateSpline(x, kdb)
                q = p.derivative()
                v = q(x)
                r = q.derivative()

                a = r(x)
                ar0, _ = find_peaks(np.abs(v), height=np.max(np.abs(v))*0.6)
                ar1, _ = find_peaks(a)
                search = []

                kdb1 = -kdb
                kdb1 = kdb1 + abs(np.min(kdb1))
                ar2, _ = find_peaks(kdb1, height=np.max(np.abs(kdb1))*0.85)
                toe_off_line = 0
                # if len(ar0) > 0:
                    # ret = None
                    # for b in ar1:
                    #     if b < ar0[0]:
                    #         search.append(b)
                    # k0 = 0
                    # if len(search) > 0:
                    #     if '2392' in model_name.lower():
                    #         k0 = int(search[-1] + (ar0[0] - search[-1]) / 2.0)
                    #         toe_off[keyw].append((h - st) + k0)
                    #     else:
                    #         mx = np.max(np.abs(a[search]))
                    #         idx = search[0]
                    #         for i in range(0, len(search)):
                    #             if (v[idx] - mx) > 1e-8:
                    #                 idx = i
                    #         k0 = int(idx+(ar0[0]-idx)/2.0)
                    #         # k0 = int(idx + (mx - idx) / 2.0)
                    #         ret = (h-st)+k0
                    #         toe_off[keyw].append(ret)
                if len(ar2) > 0:
                    kto = 0
                    ar_h = 1000
                    for ax in ar2:
                        if ax < st:
                            continue
                        if ar_h > kdb[ax]:
                            ar_h = kdb[ax]
                        else:
                            continue

                        ktop = [frame for frame in ar1 if (frame-ax) > 0]
                        kton = [frame for frame in ar1 if (frame - ax) <= 0]
                        if len(ktop) == 0:
                            ktop = [0]
                        if len(kton) == 0:
                            kton = [0]

                        if np.abs(ktop[0]-ax) >  np.abs(kton[-1]-ax):
                            kto = kton[-1]
                            if kton[-1] >= kdb.shape[0]:
                                kto = kdb.shape[0]-1
                        else:
                            kto = ktop[0]

                    ret = h-st + kto
                    toe_off_line = kto
                    toe_off[keyw].append(ret)
                if is_debug:
                    plt.figure()
                    plt.plot(np.abs(v), label='v')
                    plt.plot(a, label='a')
                    plt.plot(x, kdb, label="p")
                    plt.plot(x[ar2], kdb[ar2], 'd', label="px")
                    plt.plot(x[ar0], np.abs(v[ar0]), 'x', label='xv')
                    plt.plot(x[ar1], np.abs(a[ar1]), 'o',label='xa')
                    plt.axvline(x=st, ls='--', color='purple', label=keyw + '_search')
                    plt.axvline(x=toe_off_line, ls='--', color='red', label=keyw + '_toe_off')
                    plt.legend()
                    plt.show()
                    pass
                pass

            for h in heel[keyw]:
                print("swing {0}".format(h))
                st = 80
                if np.isnan(h):
                    continue
                if st > h:
                    st = h
                ent = h+st+30
                if ent > heel_marker[keyw].shape[0]:
                    ent = heel_marker[keyw].shape[0]-1
                ys = heel_marker[keyw].iloc[h-st:ent, :].to_numpy()
                y = [np.linalg.norm(ys[yx, up]) for yx in range(0, ys.shape[0])]
                y = Butterworth.butter_low_filter(y, 6, 1 / dt)
                arX, _ = find_peaks(y, height=np.max(y)*0.9)
                if len(arX) == 2:
                    ranger = ent-(h-st)
                    diff = ranger - arX[-1]
                    ent = ent - diff + int(diff*0.25)
                    print("diff = {0}".format(diff))
                    print("diff% = {0}".format(diff*0.25))
                elif len(arX) > 2:
                    if is_debug:
                        plt.figure()
                        plt.plot(y)
                        x = np.array([i for i in range(0, len(y))])
                        yx = np.array(y)
                        plt.plot(x[arX], yx[arX], 'x')
                        plt.show()

                ys = heel_marker[keyw].iloc[h-st:ent, :].to_numpy()
                y = [np.linalg.norm(ys[yx, up]) for yx in range(0, ys.shape[0])]
                y0 = [c/np.max(np.array(y)) for c in y]
                kdb = Butterworth.butter_low_filter(y0, 8, 1 / dt)
                x = [i for i in range(0, len(y))]
                p = interpolate.InterpolatedUnivariateSpline(x, kdb)
                q = p.derivative()
                r = q.derivative()

                a = r(x)
                ar0, _ = find_peaks(np.abs(a))
                search = []
                for b in ar0:
                    if b > st:
                        search.append(b)
                if len(search) == 0:
                    continue

                kdb1 = -kdb
                kdb1 = kdb1 + abs(np.min(kdb1))
                ar2, _ = find_peaks(kdb1, height=np.max(np.abs(kdb1))*0.85)
                if len(ar2) == 0:
                    k1 = search[-1]
                    heel_strike[keyw].append((h-st)+k1)
                else:
                    kto = kdb.shape[0]-1
                    anchor = kto
                    for ax in ar2:
                        if ax < st:
                            continue
                        ktop = [frame for frame in ar0 if (frame - ax) > 0]
                        if len(ktop) == 0:
                            ktop = [0]

                        kton = [frame for frame in ar0 if (frame - ax) <= 0]
                        if len(kton) == 0:
                            kton = [0]

                        if np.abs(ktop[0] - ax) > np.abs(kton[-1] - ax):
                            if kdb[kto] > kdb[kton[-1]]:
                                kto = kton[-1]
                        else:
                            if kdb[kto] > kdb[ktop[0]]:
                                kto = ktop[0]

                        # if anchor == kto:
                        #     print(kdb[kton[-1]])
                        #     print(kdb[ktop[0]])

                    ret = h - st + kto
                    k1 = kto
                    heel_strike[keyw].append(ret)
                print()
                print((h-st)+k1+frame_number_st)
                xi = np.array([i for i in range(0, len(x))])
                if is_debug:
                    plt.figure()
                    plt.plot(x, a, label='a')
                    plt.plot(x, kdb, label="p")
                    plt.plot(xi[ar0], a[ar0], 'o',label='ax')
                    plt.plot(xi[ar2], kdb[ar2], 'd', label='px')
                    plt.axvline(x=st, ls='--', color='purple', label=keyw + '_search')
                    plt.axvline(x=k1, ls='--', color='red', label=keyw + '_heel')
                    plt.legend()
                    plt.show()
                    pass
            pass

        if is_debug or show_final:
            plt.figure()
            cfm = plt.get_current_fig_manager()
            cfm.window.showMaximized()
            plt.title("Model::{0} -> Trial::{1}".format(modelfilename[1], ikfilename[1]))
            k = "ankle_angle_r"
            kd2 = np.array(sto.data[k].to_list())
            xi = [i+frame_number_st for i in range(0, len(kd2))]
            for keyw in ["knee_angle_r"]:
                kd = np.array(sto.data[keyw].to_list())
                kdb = Butterworth.butter_low_filter(kd, 8, 1 / dt)

                plt.plot(xi, kdb, color='blue', label=keyw)
                #plt.plot(kd2, label=k, color='green')
                for h in heel_strike[keyw]:
                    if h >= len(xi):
                        continue
                    plt.axvline(x=xi[h], color='blue', label=keyw +'_heel_strike')
                for h in toe_off[keyw]:
                    if h >= len(xi):
                        continue
                    plt.axvline(x=xi[h], ls='--', color='blue', label=keyw +'_toe_off')
                pass
                break
            k = "ankle_angle_l"
            kd2 = np.array(sto.data[k].to_list())
            for keyw in ["knee_angle_l"]:
                kd = np.array(sto.data[keyw].to_list())
                kdb = Butterworth.butter_low_filter(kd, 8, 1 / dt)
                plt.plot(xi, kdb, color='orange', label=keyw)
                #plt.plot(kd2, label=k, color='red')
                for h in heel_strike[keyw]:
                    if h >= len(xi):
                        continue
                    plt.axvline(x=xi[h], color='orange', label=keyw +'_heel_strike')
                for h in toe_off[keyw]:
                    if h >= len(xi):
                        continue
                    plt.axvline(x=xi[h], ls='--', color='orange', label=keyw +'_toe_off')
                pass
            #     break
            plt.legend()
            plt.show()

        return {"label": leg, "heel_strike": heel_strike, "toe_off": toe_off, 'ik': sto, 'trc': marker_data}

    @staticmethod
    def process_stride_heel(leg, target_markers=None):
        if leg is None:
            return None
        if target_markers is None:
            target_markers = {'right': 'RHeel', 'left': 'LHeel'}
        sto = leg['ik']
        trc = leg['trc']
        label = [m for m in leg['label']]
        left_right = {}
        for l in label:
            if l.lower().endswith('_r'):
                left_right[l] = 'right'
            elif l.lower().endswith('_l'):
                left_right[l] = 'left'
        ret = {left_right[l0]: [] for l0 in left_right}
        ret["stance_time"] = {'left': [], 'right': []}
        ret["all_normalise"] = {'left': [], 'right': []}
        ret["left_normalise"] = {'left': [], 'right': []}
        ret["right_normalise"] = {'left': [], 'right': []}
        ret["swing_time"] = {'left': [], 'right': []}
        ret["single_support"] = {'left': [], 'right': []}
        ret["double_support"] = {'left': [], 'right': []}
        ret["double_support_instance"] = {'left': [], 'right': []}
        ret["stride_time"] = {'left': [], 'right': []}
        ret["stride_length"] = {'left': [], 'right': []}
        ret["stride_speed"] = {'left': [], 'right': []}
        ret["walking_speed"] = {'left': [], 'right': []}
        ret["step_time"] = {'left': [], 'right': []}
        ret["step_length"] = {'left': [], 'right': []}
        ret["step_speed"] = {'left': [], 'right': []}
        heel = leg['heel_strike']
        toe = leg['toe_off']
        for l in label:
            side = "right"
            if l.endswith('_l'):
                side = 'left'
            for i in range(0, len(leg['heel_strike'][l])-1):
                st = heel[l][i]
                ed = heel[l][i+1]
                if ed > sto.data.shape[0]:
                    ed = sto.data.shape[0]-1
                toeE = [k for k in toe[l] if k > st]
                other = [k for k in label if k != l]
                other_toe = [k for k in toe[other[0]] if k > st]
                other_heel = [k for k in heel[other[0]] if k > st]
                subset = sto.data.iloc[st:ed, :]
                ret[left_right[l]].append(subset)
                try:
                    df = (subset.iloc[-1, 0] - subset.iloc[0, 0]) / 101
                except IndexError:
                    continue
                times = [subset.iloc[0, 0] + df * t for t in range(0, 101)]
                subset_yd = Yatsdo(subset)
                n = subset_yd.get_samples(times)
                n[:, 0] = [t for t in range(0, 101)]
                ret["all_normalise"][side].append(n)
                ns = pd.DataFrame(data=n, columns=subset_yd.col_labels)
                ret["{0}_normalise".format(left_right[l])][side].append(ns)
                try:
                    stance = sto.data.iloc[toeE[0], 0] - sto.data.iloc[st, 0]
                    ret['stance_time'][side].append(stance)
                except IndexError:
                    pass
                try:
                    swing = sto.data.iloc[ed, 0] - sto.data.iloc[toeE[0], 0]
                    ret['swing_time'][side].append(swing)
                except IndexError:
                    pass
                try:
                    double_supp = sto.data.iloc[other_toe[0], 0] - sto.data.iloc[st, 0]
                    if double_supp < 0.5:
                        ret['double_support_instance'][side].append([sto.data.iloc[st, 0], double_supp])
                    else:
                        ret['double_support_instance'][side].append([sto.data.iloc[st, 0], np.nan])

                except IndexError:
                    ret['double_support_instance'][side].append([sto.data.iloc[st, 0], np.nan])
                    pass


                try:
                    single_supp = sto.data.iloc[toeE[0], 0] - sto.data.iloc[other_toe[0], 0]
                    ret["single_support"][side].append(single_supp)
                except IndexError:
                    ret["single_support"][side].append(np.nan)
                    pass

                stride_time = sto.data.iloc[ed, 0] - sto.data.iloc[st, 0]
                ret["stride_time"][side].append(stride_time)
                pelvis = sto.data[['pelvis_tx', 'pelvis_ty', 'pelvis_tz']].iloc[st:ed, :]
                dist = np.linalg.norm(pelvis.iloc[-1, :] - pelvis.iloc[0, :])
                ret["walking_speed"][side].append(dist/stride_time)   # based on stride
                try:
                    step_time = sto.data.iloc[other_heel[0], 0] - sto.data.iloc[st, 0]
                    ret["step_time"][side].append(step_time)
                except IndexError:
                    pass
                rheels = trc.marker_set[target_markers['right']].to_numpy()
                lheels = trc.marker_set[target_markers['left']].to_numpy()

                try:
                    rh0 = rheels[st, :]
                    rh1 = rheels[ed, :]
                    lh = lheels[other_heel[0], :]
                    hyp = np.linalg.norm(lh - rh0)
                    b = lh - rh0
                    a = rh1 - rh0

                    if l.lower().endswith('l'):
                        rh0 = lheels[st, :]
                        rh1 = lheels[ed, :]
                        lh = rheels[other_heel[0], :]
                        hyp = np.linalg.norm(lh - rh0)
                        a = lh - rh0
                        b = rh1 - rh0
                    step_length = hyp * (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                    ret["step_length"][side].append(step_length)
                    ret["step_speed"][side].append(1/step_time)
                    ret["stride_length"][side].append(np.linalg.norm(b))
                    ret["stride_speed"][side].append(1/stride_time)
                except KeyError:
                    pass
                except IndexError:
                    pass

        for s in ret['double_support_instance']['right']:
            s_time = s[0]
            for o in ret['double_support_instance']['left']:
                if o[0] > s_time:
                    ds_time = s[1] + o[1]
                    ret['double_support']['right'].append({'stride_id':s[0], 'ds_time': ds_time})
                    break
        for s in ret['double_support_instance']['left']:
            s_time = s[0]
            for o in ret['double_support_instance']['right']:
                if o[0] > s_time:
                    ds_time = s[1] + o[1]
                    ret['double_support']['left'].append({'stride_id': s[0], 'ds_time': ds_time})
                    break

            # ret['double_support_instance'][s][d] + ret['double_support_instance'][other_s][d])
            # other_s = 'left'
            # if s == 'left':
            #     other_s = 'left'
            # for d in range(0, len(ret['double_support_instance'][s])):
            #     ret['double_support'][s].append(
            #         ret['double_support_instance'][s][d] + ret['double_support_instance'][other_s][d])
        return ret

    @staticmethod
    def simple_plotter(data, joint="knee_angle"):
        x = [i for i in range(0, 101)]
        combine = []
        left = []
        right = []
        for d in data["left_normalise"]:
            combine.append(d[OsimIKLabels.left_cols.value])
            left.append(d[OsimIKLabels.left_cols.value])
        for d in data["right_normalise"]:
            combine.append(d[OsimIKLabels.right_cols.value])
            right.append(d[OsimIKLabels.right_cols.value])
        combine_np = np.dstack(combine)
        left_np = np.dstack(left)
        right_np = np.dstack(right)
        mc = pd.DataFrame(data=np.nanmean(combine_np, axis=2), columns=OsimIKLabels.cols.value)
        sc = pd.DataFrame(data=np.nanstd(combine_np, axis=2), columns=OsimIKLabels.cols.value)
        mr = pd.DataFrame(data=np.nanmean(right_np, axis=2), columns=OsimIKLabels.cols.value)
        sr = pd.DataFrame(data=np.nanstd(right_np, axis=2), columns=OsimIKLabels.cols.value)
        ml = pd.DataFrame(data=np.nanmean(left_np, axis=2), columns=OsimIKLabels.cols.value)
        sl = pd.DataFrame(data=np.nanstd(left_np, axis=2), columns=OsimIKLabels.cols.value)

        plt.figure()
        plt.title(joint)
        plt.plot(x, mr[joint], label='Right')
        plt.fill_between(x, mr[joint] - sr[joint], mr[joint] + sr[joint], alpha=0.2)
        plt.plot(x, ml[joint], label='Left')
        plt.fill_between(x, ml[joint] - sl[joint], ml[joint] + sl[joint], alpha=0.2)
        plt.plot(x, mc[joint], label='All')
        plt.fill_between(x, mc[joint] - sc[joint], mc[joint] + sc[joint], alpha=0.2)
        plt.legend()
        plt.show()
        pass
        return None


class IK:
    @staticmethod
    def run(xml_file):
        subprocess.run(["ik", "-S", xml_file])

    @staticmethod
    def run_from_c3d(wkdir="M:/test/", root="M:/Mocap/P011/New Session/",
                     trial_c3d="Straight normal 1.c3d",
                     template='M:/template/Straight normal 1.xml',
                     model=None,
                     mode=0):
        IKT.run_from_c3d(wkdir, trial_c3d, template, model, mode)

    @staticmethod
    def write_ik_setup(trial, template, model, output_motion_file, save_name):
        """
        :param trial:
        :param template:
        :param model:
        :param output_motion_file:
        :param save_name:
        """
        IKT.write_ik_setup(trial, template, model, output_motion_file, save_name)
        
    @staticmethod
    def write_ik_setup_xml(ikconfig, save_name):
        IK.write_ik_setup(ikconfig["marker_file"],
                          ikconfig["template"],
                          ikconfig["model_file"],
                          ikconfig["output_motion_file"],
                          save_name)
