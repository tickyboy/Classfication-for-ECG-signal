clear;clc;
fprintf('Loading data...\n');
tic;
load('N_dat.mat');load('L_dat.mat');load('R_dat.mat');load('V_dat.mat');
fprintf('Finished!\n');
Nb=Nb(1:5000,:);Label1=repmat([1;0;0;0],1,5000);
Vb=Vb(1:5000,:);Label2=repmat([0;1;0;0],1,5000);
Rb=Rb(1:5000,:);Label3=repmat([0;0;1;0],1,5000);
Lb=Lb(1:5000,:);Label4=repmat([0;0;0;1],1,5000);
Data=[Nb;Vb;Rb;Lb];
Label=[Label1,Label2,Label3,Label4];
clear Nb;clear Label1;
clear Rb;clear Label2;
clear Lb;clear Label3;
clear Vb;clear Label4;
Data=Data-repmat(mean(Data,2),1,250);
fprintf('Model training and testing...\n');
Nums=randperm(20000);
x_btrained=Data(Nums(1:10000),:);
x_test=Data(Nums(10001:end),:);
y_btrained=Label(:,Nums(1:10000));
y_test=Label(:,Nums(10001:end));
x_btrained=x_btrained';
x_test=x_test';

cnn.layers = {
    struct('type', 'i') 
    struct('type', 'c', 'outputmaps', 4, 'kernelsize', 31,'actv','relu')
    struct('type', 's', 'scale', 5,'pool','mean') 
    struct('type', 'c', 'outputmaps', 8, 'kernelsize', 6,'actv','relu')
    struct('type', 's', 'scale', 3,'pool','mean')
};
cnn.output = 'softmax';        
opts.alpha = 0.01;
opts.batchsize = 16;     
opts.numepochs = 30;     

cnn = cnnsetup1d(cnn, x_btrained, y_btrained);
cnn = cnntrain1d(cnn, x_btrained, y_btrained,opts);
[er,bad,out] = cnntest1d(cnn, x_test, y_test);

[~,ptest]=max(out,[],1);
[~,test_yt]=max(y_test,[],1);

Correct_Predict=zeros(1,4);       
Class_Num=zeros(1,4);
Conf_Mat=zeros(4);
for i=1:10000
    Class_Num(test_yt(i))= Class_Num(test_yt(i))+1;
    Conf_Mat(test_yt(i),ptest(i))=Conf_Mat(test_yt(i),ptest(i))+1;
    if ptest(i)==test_yt(i)
        Correct_Predict(test_yt(i))= Correct_Predict(test_yt(i))+1;
    end
end

ACCs=Correct_Predict./Class_Num;
fprintf('Accuracy = %.2f%%\n',(1-er)*100);
fprintf('Accuracy_N = %.2f%%\n',ACCs(1)*100);
fprintf('Accuracy_V = %.2f%%\n',ACCs(2)*100);
fprintf('Accuracy_R = %.2f%%\n',ACCs(3)*100);
fprintf('Accuracy_L = %.2f%%\n',ACCs(4)*100);