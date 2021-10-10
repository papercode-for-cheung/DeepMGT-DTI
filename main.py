# import numpy and pandas
import numpy as np
import pandas as pd


# import keras modules
import tensorflow as tf
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Embedding, Lambda,Reshape,Flatten
from keras.layers import Convolution1D, GlobalMaxPooling1D, SpatialDropout1D,GlobalAveragePooling1D
from keras.layers import Concatenate
from keras.optimizers import Adam
from keras.regularizers import l2,l1
from keras.preprocessing import sequence


from sklearn.metrics import precision_recall_curve, auc, roc_curve

from layers.graph import GraphLayer,GraphConv
import tensorflow_transformer as tf_transformer
from sklearn.metrics import confusion_matrix
import os
import hickle as hkl
import scipy.sparse as sp
import numpy.random as random
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

seq_rdic = ['A','I','L','V','F','W','Y','N','C','Q','M','S','T','D','E','R','H','K','G','P','O','U','X','B','Z']
seq_dic = {w: i+1 for i,w in enumerate(seq_rdic)}


def encodeSeq(seq, seq_dic):
    if pd.isnull(seq):
        return [0]
    else:
        return [seq_dic[aa] for aa in seq]



def load_drug_features(training_compound_feature_file):
    # load drug features
    training_compound_drug_pubchem_id_set = []
    training_compound_drug_feature = {}
    for each in os.listdir(training_compound_feature_file):
        training_compound_drug_pubchem_id_set.append(each.split('.')[0])
        feat_mat,adj_list,degree_list = hkl.load('%s/%s'%(training_compound_feature_file,each))
        training_compound_drug_feature[each.split('.')[0]] = [feat_mat,adj_list,degree_list]
        assert len(training_compound_drug_pubchem_id_set)==len(training_compound_drug_feature.values())
    return training_compound_drug_pubchem_id_set,training_compound_drug_feature

def NormalizeAdj(adj):
    adj = adj + np.eye(adj.shape[0])
    d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten(), 0).toarray()
    a_norm = adj.dot(d).transpose().dot(d)
    return a_norm
def random_adjacency_matrix(n):
    matrix = [[random.randint(0, 1) for i in range(n)] for j in range(n)]
    # No vertex connects to itself
    for i in range(n):
        matrix[i][i] = 0
    # If i is connected to j, j is connected to i
    for i in range(n):
        for j in range(n):
            matrix[j][i] = matrix[i][j]
    return matrix
#drug 补全
def CalculateGraphFeat(feat_mat,adj_list):
    assert feat_mat.shape[0] == len(adj_list)
    feat = np.zeros((Max_atoms,feat_mat.shape[-1]),dtype='float32')
    adj_mat = np.zeros((Max_atoms,Max_atoms),dtype='float32')
    if israndom:
        feat = np.random.rand(Max_atoms,feat_mat.shape[-1])
        adj_mat[feat_mat.shape[0]:,feat_mat.shape[0]:] = random_adjacency_matrix(Max_atoms-feat_mat.shape[0])
    feat[:feat_mat.shape[0],:] = feat_mat
    for i in range(len(adj_list)):
        nodes = adj_list[i]
        for each in nodes:
            adj_mat[i,int(each)] = 1
    assert np.allclose(adj_mat,adj_mat.T)
    adj_ = adj_mat[:len(adj_list),:len(adj_list)]
    adj_2 = adj_mat[len(adj_list):,len(adj_list):]
    norm_adj_ = NormalizeAdj(adj_)
    norm_adj_2 = NormalizeAdj(adj_2)
    adj_mat[:len(adj_list),:len(adj_list)] = norm_adj_
    adj_mat[len(adj_list):,len(adj_list):] = norm_adj_2
    return [feat,adj_mat]
def calculate_graph(id_set,drug_feature):
    compound_drug_data ={}
    for pubchem_id in id_set:
        feat_mat,adj_list,_ = drug_feature[str(pubchem_id)]
        compound_drug_data[pubchem_id] = CalculateGraphFeat(feat_mat,adj_list)
    return compound_drug_data

def label_by_th(y_pred, threshold=0.5):
    new_pred = []
    for item in y_pred:
        if item>= threshold:
            new_pred.append(1)
        else:
            new_pred.append(0)
    return np.array(new_pred)
def gcn_extract(id_set,drug_data):#处理成gcn可以运行的数据
    #print(X_drug_data_train)
    drug_graph_data = {}
    for pubchem_id in id_set:
        item_drug_data = drug_data[pubchem_id]
        X_drug_feat_data_train = item_drug_data[0]
        X_drug_adj_data_train =  item_drug_data[1]
        X_drug_feat_data_train = np.array(X_drug_feat_data_train)#nb_instance * Max_stom * feat_dim
        X_drug_adj_data_train = np.array(X_drug_adj_data_train)#nb_instance * Max_stom * Max_stom
        calculated_drug_data = {'feat':X_drug_feat_data_train,'adj':X_drug_adj_data_train}
        drug_graph_data[pubchem_id] =calculated_drug_data


    return drug_graph_data

def get_compound():
    compound_data= ''
    return compound_data

def parse_data(dti_dir, drug_dir, protein_dir, with_label=True,
               prot_len=2500, prot_vec="Convolution",
               drug_vec="Convolution", mode="Train",drug_len=2048):


    print("Parsing {0},{1},{2} with length {3}, type {4}".format(*[dti_dir ,drug_dir, protein_dir, prot_len, prot_vec]))
    protein_col = "Protein_ID"
    drug_col = "Compound_ID"
    col_names = [protein_col, drug_col]
    if with_label:
        label_col = "Label"
        col_names += [label_col]
    dti_df = pd.read_csv(dti_dir)
    drug_df = pd.read_csv(drug_dir, index_col="Compound_ID")
    protein_df = pd.read_csv(protein_dir, index_col="Protein_ID")


    if prot_vec == "Convolution":
        protein_df["encoded_sequence"] = protein_df.Sequence.map(lambda a: encodeSeq(a, seq_dic))
    dti_df = pd.merge(dti_df, protein_df, left_on=protein_col, right_index=True)
    dti_df = pd.merge(dti_df, drug_df, left_on=drug_col, right_index=True)

    drug_feat_data = []
    drug_adj_data  = []


    if mode =="Train":
        print("Train")

        for pubchemid in dti_df['Compound_ID'].values:
            drug_feat_data.append(train_drug_data[str(pubchemid)]['feat'])
            drug_adj_data.append(train_drug_data[str(pubchemid)]['adj'])
    else:
        print("valid")

        for pubchemid in dti_df['Compound_ID'].values:
            drug_feat_data.append(validation_drug_data[str(pubchemid)]['feat'])
            drug_adj_data.append(validation_drug_data[str(pubchemid)]['adj'])

    #print(drug_feature)
    #print(drug_feature.shape)
    if prot_vec=="Convolution":
        protein_feature = sequence.pad_sequences(dti_df["encoded_sequence"].values, prot_len)
    else:
        protein_feature = np.stack(dti_df[prot_vec].map(lambda fp: fp.split("\t")))
    #print(protein_feature.shape,drug_feature.shape)

    if with_label:
        label = dti_df[label_col].values
        print("\tPositive data : %d" %(sum(dti_df[label_col])))
        print("\tNegative data : %d" %(dti_df.shape[0] - sum(dti_df[label_col])))
        return {"protein_feature": protein_feature, "drug_feat": drug_feat_data,"drug_adj":drug_adj_data,"label": label}
    else:
        return {"protein_feature": protein_feature, "drug_feat": drug_feat_data,"drug_adj":drug_adj_data}



class Drug_Target_Prediction(object):

    def PLayer(self, size, filters, activation, initializer, regularizer_param):
        def f(input):
            # model_p = Convolution1D(filters=filters, kernel_size=size, padding='valid', activity_regularizer=l2(regularizer_param), kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = Convolution1D(filters=filters, kernel_size=size, padding='same', kernel_initializer=initializer, kernel_regularizer=l2(regularizer_param))(input)
            model_p = BatchNormalization()(model_p)
            model_p = Activation(activation)(model_p)
            return GlobalMaxPooling1D()(model_p)
        return f

    def modelv(self, dropout, drug_layers, protein_strides, filters, fc_layers, prot_vec=False, prot_len=2500,
               activation='relu', protein_layers=None, initializer="glorot_normal", drug_len=2048, drug_vec="ECFP4"):
        def return_tuple(value):
            if type(value) is int:
               return [value]
            else:
               return tuple(value)

        regularizer_param = 0.001

        #模型输入
        #input_d = Input(shape=(drug_len,))
        drug_feat_input = Input(shape=(None,self.drug_dim),name='drug_feat_input')#drug_dim=75
        drug_adj_input = Input(shape=(None,None),name='drug_adj_input')
        input_p = Input(shape=(prot_len,))

        params_dic = {"kernel_initializer": initializer,
                      # "activity_regularizer": l2(regularizer_param),
                      "kernel_regularizer": l2(regularizer_param),
        }

        """gcn部分"""


        #drug feature with GCN
        GCN_layer = GraphConv(units=self.units_list[0],step_num=1)([drug_feat_input,drug_adj_input])

        origin_gcn_layer = GCN_layer  # 拿到第一级

        if self.use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if self.use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)

        # for i in range(len(self.units_list)-1):
        #     GCN_layer = GraphConv(units=self.units_list[i+1],step_num=1)([GCN_layer,drug_adj_input])
        #     if self.use_relu:
        #         GCN_layer = Activation('relu')(GCN_layer)
        #     else:
        #         GCN_layer = Activation('tanh')(GCN_layer)
        #     if self.use_bn:
        #         GCN_layer = BatchNormalization()(GCN_layer)
        #     GCN_layer = Dropout(0.1)(GCN_layer)

        GCN_layer = GraphConv(units=100,step_num=1)([GCN_layer,drug_adj_input])

        origin_gcn_layer_2 = GCN_layer  # 拿到第二级

        if self.use_relu:
            GCN_layer = Activation('relu')(GCN_layer)
        else:
            GCN_layer = Activation('tanh')(GCN_layer)
        if self.use_bn:
            GCN_layer = BatchNormalization()(GCN_layer)
        GCN_layer = Dropout(0.1)(GCN_layer)
        #global pooling
        if self.use_GMP:
            x_drug = GlobalMaxPooling1D()(GCN_layer)
        else:
            x_drug = GlobalAveragePooling1D()(GCN_layer)
        """gcn部分"""

        #x_drug = tf_transformer.EncoderLayer(100, 4, 512, 0.1)(x_drug, True, None)
        # model.add(Reshape((-1, 100)))
        # env_expand = Reshape((-1,100))
        # x_drug = env_expand(x_drug)

        x_drug_embedding = x_drug#tf_transformer.EncoderLayer(100, 512, 4, 0.1)(x_drug)#拿到embedding
        x_drug_merged = tf_transformer.EncoderLayer_modify(356, 1024, 6, 0.1)(origin_gcn_layer, origin_gcn_layer_2)
        # x_drug_merged = Dropout(0.1)(x_drug_merged)

        x_drug_merged = GlobalAveragePooling1D()(x_drug_merged)#
        x_drug = Concatenate()([x_drug_merged, x_drug_embedding])

        env_expand = Reshape([456])
        x_drug = env_expand(x_drug)
        model_d = x_drug

        print(model_d.shape)


        if prot_vec == "Convolution":
            model_p = Embedding(26,20, embeddings_initializer=initializer,embeddings_regularizer=l2(regularizer_param))(input_p)
            model_p = SpatialDropout1D(0.2)(model_p)
            model_ps = [self.PLayer(stride_size, filters, activation, initializer, regularizer_param)(model_p) for stride_size in protein_strides]
            if len(model_ps)!=1:
                model_p = Concatenate(axis=1)(model_ps)
            else:
                model_p = model_ps[0]
        else:
            model_p = input_p

        if protein_layers:
            input_layer_p = model_p
            protein_layers = return_tuple(protein_layers)
            for protein_layer in protein_layers:
                model_p = Dense(protein_layer, **params_dic)(input_layer_p)
                model_p = BatchNormalization()(model_p)
                model_p = Activation(activation)(model_p)

                model_p = Dropout(dropout)(model_p)
                input_layer_p = model_p

        # print(model_p.shape)
        # protein_expand = Reshape((1,128))
        # model_p = protein_expand(model_p)
        #
        print(model_p.shape)
        model_t = Concatenate(axis=1)([model_d,model_p])
        #input_dim = filters*len(protein_strides) + drug_layers[-1]
        if fc_layers is not None:
            fc_layers = return_tuple(fc_layers)
            for fc_layer in fc_layers:
                model_t = Dense(units=fc_layer,#, input_dim=input_dim,
                                **params_dic)(model_t)
                model_t = BatchNormalization()(model_t)
                model_t = Activation(activation)(model_t)
                # model_t = Dropout(dropout)(model_t)
                input_dim = fc_layer
        model_t = Dense(1, activation='tanh', activity_regularizer=l2(regularizer_param),**params_dic)(model_t)
        model_t = Lambda(lambda x: (x+1.)/2.)(model_t)
        # embed = model_t
        model_f = Model(inputs=[drug_feat_input, drug_adj_input, input_p], outputs = model_t)

        return model_f
        # return model_f, model_t

    def __init__(self, dropout=0.2, drug_layers=(2048,512), protein_strides = (10,15,20,25), filters=64,
                 learning_rate=1e-3, decay=0.0, fc_layers=None, prot_vec=None, prot_len=2500, activation="relu",
                 drug_len=2048, drug_vec="ECFP4", protein_layers=None):
        """"gcn部分参数"""
        self.drug_dim = 75
        self.use_relu=True
        self.use_bn=True
        self.use_GMP=True
        self.units_list = [256,256,256]
        """"gcn部分参数"""

        self.__dropout = dropout
        self.__drugs_layer = drug_layers
        self.__protein_strides = protein_strides
        self.__filters = filters
        self.__fc_layers = fc_layers
        self.__learning_rate = learning_rate
        self.__prot_vec = prot_vec
        self.__prot_len = prot_len
        self.__drug_vec = drug_vec
        self.__drug_len = drug_len
        self.__activation = activation
        self.__protein_layers = protein_layers
        self.__decay = decay
        self.__model_t = self.modelv(self.__dropout, self.__drugs_layer, self.__protein_strides,
                                     self.__filters, self.__fc_layers, prot_vec=self.__prot_vec,
                                     prot_len=self.__prot_len, activation=self.__activation,
                                     protein_layers=self.__protein_layers, drug_vec=self.__drug_vec,
                                     drug_len=self.__drug_len)

        opt = Adam(lr=learning_rate, decay=self.__decay)
        self.__model_t.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        K.get_session().run(tf.global_variables_initializer())

    def fit(self, protein_feature, label, n_epoch=10, batch_size=32):

        for _ in range(n_epoch):
            #print(drug_feature)
            history = self.__model_t.fit([np.random.rand(protein_feature.shape[0],100,75),np.random.rand(protein_feature.shape[0],100,100),protein_feature],label, epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1,initial_epoch=_)
        return self.__model_t
    
    def summary(self):
        self.__model_t.summary()
    
    def validation(self,protein_feature,drug_feat,drug_adj, label, output_file=None, n_epoch=10, batch_size=32, **kwargs):
        #print(kwargs.keys())
        #print(kwargs['drug_feature'].shape)
        drug_feat = np.array(drug_feat)
        drug_adj  = np.array(drug_adj)

        if output_file:
            param_tuple = pd.MultiIndex.from_tuples([("parameter", param) for param in ["window_sizes", "drug_layers", "fc_layers", "learning_rate"]])
            result_df = pd.DataFrame(data = [[self.__protein_strides, self.__drugs_layer, self.__fc_layers, self.__learning_rate]]*n_epoch, columns=param_tuple)
            result_df["epoch"] = range(1,n_epoch+1)
        result_dic = {dataset: {
            "AUC": [],
            "AUPR": [],
            "opt_threshold(AUPR)": [],
            "opt_threshold(AUC)": [],
            "sen": [],
            "pre": [],
            "spe": [],
            "acc": [],
            "f1": [],
            "loss": [],
            "accuracy": []
        } for dataset in kwargs}

        for _ in range(n_epoch):
            history = self.__model_t.fit([drug_feat,drug_adj,protein_feature],label,epochs=_+1, batch_size=batch_size, shuffle=True, verbose=1, initial_epoch=_)

            for dataset in kwargs:
                print("\tPredction of " + dataset)
                test_feat = kwargs[dataset]['drug_feat']
                test_adj  = kwargs[dataset]['drug_adj']
                test_p    = kwargs[dataset]["protein_feature"]
                test_label = kwargs[dataset]["label"]
                # print(self.__model_t)
                prediction = self.__model_t.predict([np.array(test_feat),np.array(test_adj),np.array(test_p)])
                print(prediction)
                print("#############################")
                print(test_label)
                fpr, tpr, thresholds_AUC = roc_curve(test_label, prediction)
                AUC = auc(fpr, tpr)
                precision, recall, thresholds = precision_recall_curve(test_label,prediction)
                distance = (1-fpr)**2+(1-tpr)**2

                predicted = label_by_th(prediction)
                tn, fp, fn, tp = confusion_matrix(test_label,predicted).ravel()

                sen = float(tp) / (fn + tp)
                pre = float(tp) / (tp + fp)
                spe = float(tn) / (tn + fp)
                acc = float(tn + tp) / (tn + fp + fn + tp)
                f1 = (2 * sen * pre) / (sen + pre)
                print("Evaluation of the %s set " % dataset)
                print("************************************************************************************")
                # print(vectors)

                EERs = (1-recall)/(1-precision)
                positive = sum(test_label)
                negative = test_label.shape[0]-positive
                ratio = negative/positive
                opt_t_AUC = thresholds_AUC[np.argmin(distance)]
                opt_t_AUPR = thresholds[np.argmin(np.abs(EERs-ratio))]
                AUPR = auc(recall,precision)
                print("\tArea Under ROC Curve(AUC): %0.3f" % AUC)
                print("\tArea Under PR Curve(AUPR): %0.3f" % AUPR)
                print("\tOptimal threshold(AUC)   : %0.3f " % opt_t_AUC)
                print("\tOptimal threshold(AUPR)   : %0.3f " % opt_t_AUPR)
                print("\tsen                      : %0.3f" % sen)
                print("\tpre                      : %0.3f" % pre)
                print("\tspe                      : %0.3f" % spe)
                print("\tacc                      : %0.3f " % acc)
                print("\tf1                       : %0.3f" % f1)
                print("=================================================")
                result_dic[dataset]["AUC"].append(AUC)
                result_dic[dataset]["AUPR"].append(AUPR)
                result_dic[dataset]["opt_threshold(AUC)"].append(opt_t_AUC)
                result_dic[dataset]["opt_threshold(AUPR)"].append(opt_t_AUPR)

                result_dic[dataset]["sen"].append(sen)
                result_dic[dataset]["pre"].append(pre)
                result_dic[dataset]["spe"].append(spe)
                result_dic[dataset]["acc"].append(acc)
                result_dic[dataset]["f1"].append(f1)
                result_dic[dataset]["loss"].append(history.history['loss'])
                result_dic[dataset]["accuracy"].append(history.history['accuracy'])
        if output_file:
            for dataset in kwargs:
                result_df[dataset, "AUC"] = result_dic[dataset]["AUC"]
                result_df[dataset, "AUPR"] = result_dic[dataset]["AUPR"]
                result_df[dataset, "opt_threshold(AUC)"] = result_dic[dataset]["opt_threshold(AUC)"]
                result_df[dataset, "opt_threshold(AUPR)"] = result_dic[dataset]["opt_threshold(AUPR)"]

                result_df[dataset, "sen"] = result_dic[dataset]["sen"]
                result_df[dataset, "pre"] = result_dic[dataset]["pre"]
                result_df[dataset, "spe"] = result_dic[dataset]["spe"]
                result_df[dataset, "acc"] = result_dic[dataset]["acc"]
                result_df[dataset, "f1"] = result_dic[dataset]["f1"]

                result_df[dataset, "loss"] = result_dic[dataset]["loss"]
                result_df[dataset, "accuracy"] = result_dic[dataset]["accuracy"]

            print("save to " + output_file)

            print(result_df)
            result_df.to_csv(output_file, index=False)

    def predict(self, **kwargs):
        results_dic = {}
        for dataset in kwargs:
            result_dic = {}
            test_p = kwargs[dataset]["protein_feature"]
            test_d = kwargs[dataset]["drug_feature"]
            result_dic["label"] = kwargs[dataset]["label"]
            result_dic["predicted"] = self.__model_t.predict([test_d, test_p])
            results_dic[dataset] = result_dic
        return results_dic
    
    def save(self, output_file):
        self.__model_t.save(output_file)


if __name__ == '__main__':
    training_compound_feature_file =
    validation_compound_feature_file =
    Max_atoms = 150
    israndom = False



    #
    import argparse
    parser = argparse.ArgumentParser(description="None")
    # train_params
    parser.add_argument("dti_dir", help="Training DTI information [drug, target, label]")
    parser.add_argument("drug_dir", help="Training drug information [drug, SMILES,[feature_name, ..]]")
    parser.add_argument("protein_dir", help="Training protein information [protein, seq, [feature_name]]")
    # test_params
    parser.add_argument("--test-name", '-n', help="Name of test data sets", nargs="*")
    parser.add_argument("--test-dti-dir", "-i", help="Test dti [drug, target, [label]]", nargs="*")
    parser.add_argument("--test-drug-dir", "-d", help="Test drug information [drug, SMILES,[feature_name, ..]]", nargs="*")
    parser.add_argument("--test-protein-dir", '-t', help="Test Protein information [protein, seq, [feature_name]]", nargs="*")
    parser.add_argument("--with-label", "-W", help="Existence of label information in test DTI", action="store_true")
    # structure_params
    parser.add_argument("--window-sizes", '-w', help="Window sizes for model (only works for Convolution)", default=None, nargs="*", type=int)
    parser.add_argument("--protein-layers","-p", help="Dense layers for protein", default=None, nargs="*", type=int)
    parser.add_argument("--drug-layers", '-c', help="Dense layers for drugs", default=None, nargs="*", type=int)
    parser.add_argument("--fc-layers", '-f', help="Dense layers for concatenated layers of drug and target layer", default=None, nargs="*", type=int)
    # training_params
    parser.add_argument("--learning-rate", '-r', help="Learning late for training", default=1e-4, type=float)
    parser.add_argument("--n-epoch", '-e', help="The number of epochs for training or validation", type=int, default=10)
    # type_params
    parser.add_argument("--prot-vec", "-v", help="Type of protein feature, if Convolution, it will execute conlvolution on sequeunce", type=str, default="Convolution")
    parser.add_argument("--prot-len", "-l", help="Protein vector length", default=2500, type=int)

    parser.add_argument("--drug-vec", "-V", help="Type of drug feature", type=str, default="None")
    parser.add_argument("--drug-len", "-L", help="Drug vector length", default=100, type=int)
    # the other hyper-parameters
    parser.add_argument("--activation", "-a", help='Activation function of model', type=str)
    parser.add_argument("--dropout", "-D", help="Dropout ratio", default=0.2, type=float)
    parser.add_argument("--n-filters", "-F", help="Number of filters for convolution layer, only works for Convolution", default=64, type=int)
    parser.add_argument("--batch-size", "-b", help="Batch size", default=32, type=int)
    parser.add_argument("--decay", "-y", help="Learning rate decay", default=0.0, type=float)
    # mode_params
    parser.add_argument("--validation", help="Excute validation with independent data, will give AUC and AUPR (No prediction result)", action="store_true")
    parser.add_argument("--predict", help="Predict interactions of independent test set", action="store_true")
    # output_params
    parser.add_argument("--save-model", "-m", help="save model", type=str)
    parser.add_argument("--output", "-o", help="Prediction output", type=str)

    args = parser.parse_args()
    # train data
    train_dic = {
        "dti_dir": args.dti_dir,
        "drug_dir": args.drug_dir,
        "protein_dir": args.protein_dir,
        "with_label": True,
        "mode":"Train"
    }
    # create dictionary of test_data
    test_names = args.test_name
    tests = args.test_dti_dir
    test_proteins = args.test_protein_dir
    test_drugs = args.test_drug_dir
    test_sets = zip(test_names, tests, test_drugs, test_proteins)
    output_file = args.output
    # model_structure variables
    drug_layers = args.drug_layers
    window_sizes = args.window_sizes
    if window_sizes==0:
        window_sizes = None
    protein_layers = args.protein_layers
    fc_layers = args.fc_layers
    # training parameter
    train_params = {
        "n_epoch": args.n_epoch,
        "batch_size": args.batch_size,
    }
    # type parameter
    type_params = {
        "prot_vec": args.prot_vec,
        "prot_len": args.prot_len,
        "drug_vec": args.drug_vec,
        "drug_len": args.drug_len,
    }
    # model parameter
    model_params = {
        "drug_layers": drug_layers,
        "protein_strides": window_sizes,
        "protein_layers": protein_layers,
        "fc_layers": fc_layers,
        "learning_rate": args.learning_rate,
        "decay": args.decay,
        "activation": args.activation,
        "filters": args.n_filters,
        "dropout": args.dropout
    }

    model_params.update(type_params)
    print("\tmodel parameters summary\t")
    print("=====================================================")
    for key in model_params.keys():
        print("{:20s} : {:10s}".format(key, str(model_params[key])))
    print("=====================================================")

    dti_prediction_model = Drug_Target_Prediction(**model_params)
    dti_prediction_model.summary()

    training_compound_drug_pubchem_id_set,training_compound_drug_feature = load_drug_features(training_compound_feature_file)
    #print(training_compound_drug_feature)
    validation_compound_drug_pubchem_id_set,validation_compound_drug_feature = load_drug_features(validation_compound_feature_file)
    #print(len(training_compound_drug_feature),len(validation_compound_drug_feature))
    training_compound_drug_data = calculate_graph(training_compound_drug_pubchem_id_set,training_compound_drug_feature)
    validation_compound_drug_data = calculate_graph(validation_compound_drug_pubchem_id_set,validation_compound_drug_feature)

    train_drug_data = gcn_extract(training_compound_drug_pubchem_id_set,training_compound_drug_data)
    validation_drug_data = gcn_extract(validation_compound_drug_pubchem_id_set,validation_compound_drug_data)


    # read and parse training and test data
    train_dic.update(type_params)
    train_dic = parse_data(**train_dic)
    test_dic = {test_name: parse_data(test_dti, test_drug, test_protein, mode="valid", with_label=True, **type_params) for test_name, test_dti, test_drug, test_protein in test_sets}


    # validation mode
    if args.validation:
        validation_params = {}
        validation_params.update(train_params)
        validation_params["output_file"] = output_file
        print("\tvalidation summary\t")
        print("=====================================================")
        for key in validation_params.keys():
            print("{:20s} : {:10s}".format(key, str(validation_params[key])))
        print("=====================================================")
        validation_params.update(train_dic)
        validation_params.update(test_dic)
        print(validation_params.keys())

        dti_prediction_model.validation(**validation_params)
    # prediction mode
    elif args.predict:
        print("prediction")
        train_dic.update(train_params)
        dti_prediction_model.fit(**train_dic)
        test_predicted = dti_prediction_model.predict(**test_dic)
        result_df = pd.DataFrame()
        result_columns = []
        for dataset in test_predicted:
            temp_df = pd.DataFrame()
            value = test_predicted[dataset]["predicted"]
            value = np.squeeze(value)
            print(dataset+str(value.shape))
            temp_df[dataset,'predicted'] = value
            temp_df[dataset, 'label'] = np.squeeze(test_predicted[dataset]['label'])
            result_df = pd.concat([result_df, temp_df], ignore_index=True, axis=1)
            result_columns.append((dataset, "predicted"))
            result_columns.append((dataset, "label"))
        result_df.columns = pd.MultiIndex.from_tuples(result_columns)
        print("save to %s"%output_file)
        result_df.to_csv(output_file, index=False)


    exit()
