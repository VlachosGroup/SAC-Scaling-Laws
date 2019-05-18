# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:01:06 2019

@author: wangyf
"""

m1 = np.mean(X_poly_unrepeated[:,1])
s1 = np.std(X_poly_unrepeated[:,1])

sv = Xscaler.scale_
mv = Xscaler.mean_

additional_b0 = np.sum(mv/sv)