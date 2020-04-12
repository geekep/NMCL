"""
Loads data
"""

import os
import random

import nltk
import pandas
import numpy

import torch

from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn import preprocessing

#******************** UNIMODAL REGRESSOR SEQUENCE ********************#
class UnimodalRegressorSequenceDataset(Dataset):
	def __init__(self, dataset_file_path, features_path, ids, max_len, random_crop, model_type,modality):#这里的ids指的是train的id

		label = 'PHQ_Score'

		dataset = pandas.DataFrame(pandas.read_csv(dataset_file_path))#表格型的数据结构

		self.ids = ids
		self.max_len = max_len
		self.random_crop = random_crop
		self.model_type = model_type
		self.modality = modality


		if self.modality=='vision':
			self.features = dict([(int(d[0]),os.path.join(features_path,str(int(d[0]))+'.txt')) for idx,d in dataset.iterrows() if int(d[0]) in self.ids])
		else:
			self.features = dict(
				[(int(d[0]), os.path.join(features_path, str(int(d[0])) + '.txt')) for idx, d in dataset.iterrows() if
				 int(d[0]) in self.ids])
			#self.features = dict([(int(d[0]), os.path.join(features_path, str(int(d[0])) + '.npy')) for idx, d in dataset.iterrows() if int(d[0]) in self.ids])#将0替代ids，代表的是字典id；特征路径

		self.label=dict([(int(d[0]),d[2]) for idx,d in dataset.iterrows() if int(d[0]) in self.ids])#修改，代表的是字典id:score




	def __getitem__(self, idx):
		item_id = self.ids[idx]
		if self.modality=='vision':
			item=numpy.loadtxt(self.features[item_id],delimiter=',',dtype=float, skiprows=1)
		else:
			item = numpy.load(self.features[item_id])#修改
		#length=item.shape[0]
		if item.shape[0] >self.max_len:
			length=self.max_len
			if self.random_crop:
				start_i = random.randint(0, item.shape[0] - self.max_len)
				item = item[start_i:start_i + self.max_len, :]
			else:
				start_i = int((item.shape[0] - self.max_len) / 2)
				item = item[start_i:start_i + self.max_len, :]

		item = torch.Tensor(item)
		#length=Variable(torch.Tensor(length))
		label=Variable(torch.Tensor([self.label[item_id]]))



		return item,label

	def __len__(self):
		return len(self.ids)

class UnimodalRegressorSequenceTestDataset(Dataset):
	def __init__(self, dataset_file_path, features_path, ids, max_len, random_crop, model_type,modality,transform=None):

		dataset = pandas.DataFrame(pandas.read_csv(dataset_file_path))

		self.ids = ids
		self.max_len = max_len
		self.random_crop = random_crop
		self.model_type = model_type
		self.modality = modality
		if self.modality=='vision':
			self.features = dict([(int(d[0]),os.path.join(features_path,str(int(d[0]))+'.txt')) for idx,d in dataset.iterrows() if int(d[0]) in self.ids])
		else:
			self.features = dict(
				[(int(d[0]), os.path.join(features_path, str(int(d[0])) + '.txt')) for idx, d in dataset.iterrows() if
				 int(d[0]) in self.ids])
		#self.features = dict([(int(d[0]), os.path.join(features_path, str(int(d[0])) + '.npy')) for idx, d in dataset.iterrows() if int(d[0]) in self.ids])#修改

	def __getitem__(self, idx):
		item_id = self.ids[idx]
		if self.modality=='vision':
			item=numpy.loadtxt(self.features[item_id],delimiter=',',dtype=float, skiprows=1)
		else:
			item = numpy.load(self.features[item_id])#修改
		#item = numpy.load(self.features[item_id])#修改
		if item.shape[0] > self.max_len:
			if self.random_crop:
				start_i = random.randint(0, item.shape[0] - self.max_len)
				item = item[start_i:start_i + self.max_len, :]
			else:
				start_i = int((item.shape[0] - self.max_len) / 2)
				item = item[start_i:start_i + self.max_len, :]

		item = torch.Tensor(item)

		return item

	def __len__(self):
		return len(self.ids)


def collate_fn_unimodal_regressor_sequence_dataset(data):

	original_sort = list()

	for i in range(0, len(data)):
		data[i] = (i, data[i])
	data.sort(key=lambda x: x[1][0].shape[0], reverse=True)

	for i in range(0, len(data)):
		original_sort.append(data[i][0])

	ids, tmp_features_labels = zip(*data)
	features_tmp, labels_tmp = zip(*tmp_features_labels)
	features_dim = features_tmp[0].shape[1]
	lengths = [feature.shape[0] for feature in features_tmp]

	sort_indx = numpy.argsort(original_sort)

	features = torch.zeros((len(features_tmp), max(lengths), features_dim)).float()
	for i, feature in enumerate(features_tmp):
		end = lengths[i]
		features[i, :end, :] = feature[:end, :]
	
	labels = torch.Tensor(labels_tmp).float()

	return features, lengths, labels, sort_indx

def collate_fn_unimodal_regressor_sequence_test(data):

	original_sort = list()

	for i in range(0, len(data)):
		data[i] = (i, data[i])
	data.sort(key=lambda x: x[1][0].shape[0], reverse=True)

	for i in range(0, len(data)):
		original_sort.append(data[i][0])

	ids, features_tmp = zip(*data)
	features_dim = features_tmp[0].shape[1]
	lengths = [feature.shape[0] for feature in features_tmp]

	sort_indx = numpy.argsort(original_sort)

	features = torch.zeros((len(features_tmp), max(lengths), features_dim)).float()
	for i, feature in enumerate(features_tmp):
		end = lengths[i]
		features[i, :end, :] = feature[:end, :]

	return features, lengths, sort_indx

def get_unimodal_regressor_sequence_dataset(dataset_file_path, features_path, ids, model_type,modality, batch_size, shuffle, split, max_len, workers_num, collate_fn):

	if split != 'test':
		dataset = UnimodalRegressorSequenceDataset(	dataset_file_path=dataset_file_path,
													features_path=features_path,
													ids=ids,
													max_len=max_len,
													random_crop=False,
													model_type=model_type,
													modality=modality)
	else:
		dataset = UnimodalRegressorSequenceTestDataset(	dataset_file_path=dataset_file_path,
														features_path=features_path,
														ids=ids,
														max_len=max_len,
														random_crop=False,
														model_type=model_type,
														modality=modality)
	
	data_loader = DataLoader(					dataset=dataset,
												batch_size=batch_size,#每个batch中有多少样本
												shuffle=shuffle,#指的是要不要打乱顺序
												num_workers=workers_num,
												collate_fn=collate_fn,
												pin_memory=True)


	return data_loader

def get_loaders_unimodal_regressor_sequence_dataset(ids, opt):
	features_path = os.path.join(opt.dataset_path, opt.modality, opt.feature_type)#其表明特征应该在dataset_path/modality/feature_type
	file_path = os.path.join(opt.dataset_path, opt.dataset_file_path)#dataset_file_path指的是什么呢

	train_loader = get_unimodal_regressor_sequence_dataset(	dataset_file_path=file_path,
															features_path=features_path,
															ids=ids['train'],
															model_type=opt.model_type,
															modality=opt.modality,
															batch_size=opt.batch_size,
															shuffle=True,
															split='train',
															max_len=opt.max_sequence_length,
															workers_num=opt.workers_num,
															collate_fn=collate_fn_unimodal_regressor_sequence_dataset)#collate_fn表示如何取样本样本

	val_loader = get_unimodal_regressor_sequence_dataset(	dataset_file_path=file_path,
															features_path=features_path,
															ids=ids['val'],
															model_type=opt.model_type,
															modality=opt.modality,
															batch_size=opt.batch_size,
															shuffle=False,
															split='validation',
															max_len=opt.max_sequence_length,
															workers_num=opt.workers_num,
															collate_fn=collate_fn_unimodal_regressor_sequence_dataset)
	test_loader = get_unimodal_regressor_sequence_dataset(	dataset_file_path=file_path,
															features_path=features_path,
															ids=ids['test'],
															model_type=opt.model_type,
															modality=opt.modality,
															batch_size=opt.batch_size,
															shuffle=False,
															split='test',
															max_len=opt.max_sequence_length,
															workers_num=opt.workers_num,
															collate_fn=collate_fn_unimodal_regressor_sequence_test)


	return train_loader, val_loader, test_loader


