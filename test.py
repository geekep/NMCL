import pandas as pd
import numpy as np
import os



dataset = pd.DataFrame(pd.read_csv('D:\抑郁症工作\数据集\详细数据集/all_subjects_scaled.csv'))#表格型的数据结构
for idx, d in dataset.iterrows():
    data=pd.read_csv(os.path.join('D:\抑郁症工作\数据集\详细数据集',str(int(d[0]))+'_P',str(int(d[0]))+'_TRANSCRIPT.csv'))
    np.save(os.path.join('D:\抑郁症工作\新的数据集/text\TRANSCRIPT',str(int(d[0]))+'.csv'),data)
    '''
    f = open(os.path.join('D:\抑郁症工作\数据集\详细数据集', str(int(d[0])) + '_p', str(int(d[0])) + '_CLNF_pose.txt'), 'r')
    ftext = f.read()
    f1 = open(os.path.join('D:\抑郁症工作\新的数据集/vision\CLNF_pose', str(int(d[0])) + '.txt'), 'w')
    f1.write(ftext)
    f1.close()



for i in range(4):
   f=open(os.path.join('D:\抑郁症工作\迎合数据集',str(int(300+i))+'_p',str(int(300+i))+'_CLNF_features3D.txt'),'r')
   ftext = f.read()
   f1=open(os.path.join('D:\抑郁症工作\迎合数据集/vision\CLNF_features3D',str(int(300+i))+'.txt'),'w')
   f1.write(ftext)
   f1.close()
'''