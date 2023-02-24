# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


#!/usr/bin/env python
from __future__ import print_function
import numpy as np

from collections import namedtuple
TrialInfo = namedtuple("TrialInfo", ["user", "case", 'approach_attempts', 'grasp_attempts', 'drops',
        "times_last", "times_first", "times_human"],
        verbose=False)
"""
        TrialInfo("cpaxton", "azure", 11, 6 , 4,
            times_last=[11.402739048, 15.183423042, 16.280439138, 8.687517881],
            times_first=[35.635370016, 15.183423042, 28.165201187, 45.341375828],
            times_human=[36.891602039, 15.949740887, 29.710445166, 46.218950987]),
        TrialInfo("cpaxton", "baseline", 10, 6, 4,
            times_last=[11.587481022, 10.147964001, 11.764434099, 11.645602942],
            times_first=[11.587481022, 22.55998087, 36.425435066, 11.645602942],
            times_human=[12.365466833, 23.29693985, 37.075868129, 12.328214884]),
        TrialInfo("cpaxton", "classification", 6, 4, 4,
            times_last=[14.219264031, 7.483151913, 10.617828846, 29.371556043],
            times_first=[14.219264031, 9.942044974, 10.617828846, 44.23951602],
            times_human=[14.928934098, 11.247349978, 11.300751925, 47.197077036]),
        TrialInfo("weiy", "classification", 8, 5, 5,
            times_last=[10.844556093, 9.072248936, 10.245596171, 13.438255071, 13.544636965],
            times_first=[10.844556093, 9.072248936, 10.245596171, 33.292531967, 13.544636965],
            times_human=[11.526853084, 9.715538025, 10.832019091, 33.903098106, 14.148524046]),
        TrialInfo("weiy", "baseline", 9, 6, 4,
            times_last=[11.720911026, 11.326265096, 12.583218098, 10.183686971],
            times_first=[11.720911026, 11.326265096, 20.018939972, 42.115820884],
            times_human=[12.370907069, 12.044325113, 20.757308007, 42.525731086]),
        TrialInfo("weiy", "azure", 16, 4, 4,
            times_last=[11.96747899, 10.557506085, 29.744360923, 16.984508038],
            times_first=[11.96747899, 10.557506085, 90.230449914, 53.64602685],
            times_human=[14.715355873, 11.020941973, 90.834825992, 54.307392836]),
        """
results = [
        TrialInfo("adam", "freeform", 7, 4, 4, 
            times_last=[13.189851045, 9.690932035, 10.185709953, 14.218443871],
            times_first=[13.189851045, 15.463258981, 15.920727968, 14.218443871],
            times_human=[13.941790104, 20.178209066, 16.559320927, 14.90515089],),
        TrialInfo("adam", "taught", 10, 5, 4,
            times_last=[11.331645966, 20.364185095, 10.848322153, 13.002002001],
            times_first=[11.331645966, 58.286029101, 25.787767172, 22.741108894],
            times_human=[12.406729937, 59.02609992, 26.426988125, 23.353790998],),
        TrialInfo("adam", "distracted", 10, 5, 4,
            times_last=[13.406168937, 12.205609083, 10.048871994, 13.595123053],
            times_first=[13.406168937, 12.205609083, 23.510966063, 39.011574984],
            times_human=[14.110039949, 12.856153965, 24.320401907, 42.255227089],),
        TrialInfo("brian", "freeform", 12, 6, 5,
            times_last=[10.957669974, 9.489569903, 16.369026899, 13.641510009, 13.758543015],
            times_first=[24.215105057, 14.338423968, 19.805125952, 17.17807579, 27.544891119],
            times_human=[24.957691908, 15.28323102, 20.53459692, 17.80893588, 28.285625935],),
        TrialInfo("brian", "taught", 6, 5, 4,
            times_last=[33.468163967, 26.267167092, 16.058372974, 8.368799924],
            times_first=[33.468163967, 26.267167092, 16.058372974, 25.441232919],
            times_human=[34.463197946, 26.897195101, 16.768004179, 26.122591018],),
        TrialInfo("brian", "distracted", 12, 5, 4,
            times_last=[13.084892034, 6.606208086, 6.736946821, 10.730515957],
            times_first=[13.084892034, 18.160769939, 26.18817377, 17.550153971],
            times_human=[13.846603155, 19.128207922, 27.10934782, 18.470356941],),
        TrialInfo("clemens", "freeform", 15, 8, 4,
            times_last=[13.854071855, 10.072072029, 17.9453578, 9.079376936],
            times_first=[13.854071855, 24.449423075, 53.106321812, 55.82674694],
            times_human=[14.71925497, 25.104074001, 53.737707853, 56.692203998],),
        TrialInfo("clemens", "taught", 5, 4, 4,
            times_last=[10.441082954, 9.777086974, 10.989813089, 18.739741087],
            times_first=[10.441082954, 12.43854189, 10.989813089, 18.739741087],
            times_human=[11.798368931, 13.063708783, 11.670161009, 19.386875153],),
        TrialInfo("clemens", "distracted", 7, 4, 4,
            times_last=[13.084926129, 10.285391092, 7.870623112, 6.344269991],
            times_first=[13.084926129, 10.285391092, 7.870623112, 9.853469849],
            times_human=[14.064868927, 11.01527214, 8.497284174, 10.493839026],),
        TrialInfo("martin", "freeform", 11, 8, 4,
            times_last=[16.735475063, 9.282917977, 25.854979038, 11.632247925],
            times_first=[41.152218103, 19.843087197, 25.854979038, 29.508948803],
            times_human=[42.242009163, 20.759042979, 26.783211946, 30.205271959],),
        TrialInfo("martin", "taught", 8, 4, 4,
            times_last=[10.353649139, 17.411365986, 8.320708036, 10.103767157],
            times_first=[10.353649139, 17.411365986, 10.110829115, 17.469349146],
            times_human=[11.738153934, 18.68592, 10.740427971, 18.987687111],),
        #TrialInfo("martin", "distracted", 3, 3 ,1,
        #    times_last=[15.65192914],
        #    times_first=[15.65192914],
        #    times_human=[16.398311138],),
        TrialInfo("martin", "distracted", 13, 8, 4,
            times_last=[17.999821902, 13.466383934, 15.792387008, 9.214920998],
            times_first=[27.747217894, 39.68048501, 42.411355018, 9.214920998],
            times_human=[28.775349856, 40.871285915, 43.349858045, 10.364811897],),
        TrialInfo("maru", "freeform", 6, 4, 4,
            times_last=[13.693307162, 17.258008004, 13.449814081, 12.110310793],
            times_first=[18.682834149, 17.258008004, 13.449814081, 19.609589815],
            times_human=[19.303823233, 17.927299023, 14.084251165, 20.232138872],),
        TrialInfo("maru", "taught", 8, 4, 4,
            times_last=[21.526875019, 12.677608013, 12.28533411, 14.988052845],
            times_first=[21.526875019, 12.677608013, 16.336127996, 21.364244938],
            times_human=[22.173514127, 13.303987026, 16.950922012, 22.183031798],),
        TrialInfo("maru", "distracted", 15, 4, 4,
            times_last=[16.763397932, 10.928555012, 14.439250946, 12.52073288],
            times_first=[16.763397932, 36.00387311, 27.322937965, 20.982856989],
            times_human=[17.408299923, 36.620279074, 28.059907913, 21.625782013]),
        TrialInfo("paola", "freeform", 30, 10, 4,
            times_last=[15.270796061, 9.547502994, 9.601824999, 7.748345137],
            times_first=[21.492681981, 96.245885849, 43.481809139, 15.346677065],
            times_human=[22.119894028, 97.287746906, 44.1224401, 16.023135186],),
        TrialInfo("paola", "taught", 27, 5, 4,
            times_last=[10.681361914, 11.728286982, 14.687326192, 12.062000037],
            times_first=[15.949172974, 107.54040289, 49.279064178, 33.664921999],
            times_human=[16.685850859, 108.183090926, 50.52898097, 34.715986967],),
        TrialInfo("paola", "distracted", 35, 9, 4,
            times_last=[8.839914084, 8.769407034, 13.213653088, 14.517593145],
            times_first=[64.222375155, 47.81199503, 107.695995093, 14.517593145],
            times_human=[64.981989145, 49.227588177, 109.107834101, 15.248400211],),
        TrialInfo("yash", "freeform", 11, 6, 4,
            times_last=[10.032448054, 10.767480851, 12.752690077, 10.345793009],
            times_first=[32.898524046, 28.936416864999998, 14.885565996, 27.486850977],
            times_human=[33.526664019, 29.577428818, 15.516331911, 28.226130009],),
        TrialInfo("yash", "taught", 7, 4, 4,
            times_last=[18.924075842, 11.281505108, 15.121639014, 25.238385916],
            times_first=[20.611804962, 11.281505108, 17.38884902, 27.914090872],
            times_human=[22.235779047, 12.393455028, 18.088823081, 29.392521859],),
        TrialInfo("yash", "distracted", 7, 5, 4,
            times_last=[12.502260924, 7.011761904, 8.512351036, 12.694837094],
            times_first=[12.502260924, 9.560469866, 17.846233844, 14.547496081],
            times_human=[13.98923707, 10.530781984, 18.800215959, 15.185029984],),
        TrialInfo("yu", "freeform", 9, 4, 4,
            times_last=[20.071572066, 16.889008045, 11.606631994, 19.999264956],
            times_first=[24.785309077, 18.253019094, 18.298230171, 31.287605048],
            times_human=[25.45729518, 18.885371923, 18.923314094, 31.920909167],),
        TrialInfo("yu", "taught", 5, 4, 4,
            times_last=[16.027438879, 12.711569071, 14.098745107, 20.626116037],
            times_first=[16.027438879, 12.711569071, 14.098745107, 23.776199102],
            times_human=[17.650964022, 13.344835043, 14.735469103, 24.447811127],),
        TrialInfo("yu", "distracted", 10, 8, 4,
            times_last=[13.423737049, 9.475208997, 8.714337826, 13.168983937],
            times_first=[58.976566076, 21.614151954, 20.259725809, 15.123163939],
            times_human=[59.730530977, 22.297821998, 21.013149977, 16.068250895],),
        TrialInfo("yu-wei", "freeform", 12, 5, 4,
            times_last=[12.052088976, 18.027736902, 11.368153095, 10.402108907],
            times_first=[13.262727976, 27.381249905, 25.193703174, 27.977518081],
            times_human=[14.553878069, 28.05443883, 25.811665058, 31.960432052],),
        TrialInfo("yu-wei", "taught", 14, 5, 4,
            times_last=[17.007126093, 11.293561936, 12.440615177, 13.306619168],
            times_first=[22.720757008, 27.635087014, 24.220340967, 34.033694983],
            times_human=[23.498840094, 28.481551886, 25.109233141, 35.377767086],),
        TrialInfo("yu-wei", "distracted", 12, 5, 4, #4, 2, 1,
            #times_last=[12.131860018],
            #times_human=[13.488258839]
            times_last=[8.77815485, 7.8523039820000005, 10.977282048, 5.239872932],
            times_first=[67.227912903, 12.917994022, 18.96729517, 10.911015987],
            times_human=[68.140729904, 13.84206605, 19.609367133, 11.717680931],)
            ]

# cases = "azure", "baseline", "classification"
cases = "freeform", "taught", "distracted"
by_case = {}

for trial in results:
    case = trial.case
    user = trial.user

    assert case in cases

    if case not in by_case:
        by_case[case] = {}

    if user in by_case[case]:
        print(user)
        print(by_case[case])
        raise RuntimeError('not allowed')

    by_case[case][user] = trial

for case in cases:
    count = 0
    count_times = 0
    sum_appr = 0.
    sum_grasp = 0.
    sum_drops = 0.
    sum_first = 0.
    sum_last = 0.
    sum_human = 0.
    for trial in by_case[case].values():
        count += 1
        sum_appr += trial.approach_attempts
        sum_grasp += trial.grasp_attempts
        sum_drops += trial.drops
        count_times += len(trial.times_last)
        sum_first += np.sum(trial.times_first)
        sum_last += np.sum(trial.times_last)
        sum_human += np.sum(trial.times_human)

    print ("======", case.upper(), "=======")
    print("avg approach attempts =", sum_appr / count)
    print("avg grasp attempts =", sum_grasp / count)
    print("avg drop attempts =", sum_drops / count)
    print("approach success =", sum_drops / sum_appr)
    print("grasp success =", sum_drops / sum_grasp)
    print("avg motion time =", sum_last / count_times)
    print("avg total motion time =", sum_first / count_times)
    print("avg total task time =", sum_human / count_times)

count = 0
count_times = 0
sum_appr = 0.
sum_grasp = 0.
sum_drops = 0.
sum_first = 0.
sum_last = 0.
sum_human = 0.

for case in cases:
    for trial in by_case[case].values():
        count += 1
        sum_appr += trial.approach_attempts
        sum_grasp += trial.grasp_attempts
        sum_drops += trial.drops
        count_times += len(trial.times_last)
        sum_first += np.sum(trial.times_first)
        sum_last += np.sum(trial.times_last)
        sum_human += np.sum(trial.times_human)

print ("====== ALL =======")
print("avg approach attempts =", sum_appr / count)
print("avg grasp attempts =", sum_grasp / count)
print("avg drop attempts =", sum_drops / count)
print("approach success =", sum_drops / sum_appr)
print("grasp success =", sum_drops / sum_grasp)
print("avg motion time =", sum_last / count_times)
print("avg total motion time =", sum_first / count_times)
print("avg total task time =", sum_human / count_times)


