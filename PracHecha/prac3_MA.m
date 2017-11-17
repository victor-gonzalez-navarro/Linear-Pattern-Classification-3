%Lectura de la base de datos de los MICROARRAY
clear
close all
% Lectura de etiquetas
load ref.txt
% Lectura de datos
X = textread('nci.txt','','delimiter',' '); 
X=X';
Labels=ref;
clear ref
%OPCIONES
V_clases=[1 2 3 4 6 7 9 12];            %Vector de CLASES SELECCIONADAS
                                        %son las que contienen más de dos vectores
i_dib=1;                                %0 NO /1 SI: DIBUJOS DE VECTORES
i_dist=1;                               % 1 Cálcula distancias entre elementos y entre clases
P_train=0.5;                            % Porcentaje de elementos de train                   
K_neig=input('valor de K_neig = ');     % PARAMETRO K en knnc
N_feat=6830;
%% Database partition
Index_train=[];
Index_test=[];
for i_class=1:length(V_clases);
    index=find(Labels==V_clases(i_class));
    N_i_class=length(index);
    [I_train,I_test] = dividerand(N_i_class,P_train,1-P_train);
    Index_train=[Index_train;index(I_train)];
    Index_test=[Index_test;index(I_test)];
end
% Train Selection and mixing
X_train=X(Index_train,:);
Labels_train=Labels(Index_train);
Index_train=randperm(length(Labels_train));
X_train=X_train(Index_train,:);
Labels_train=Labels_train(Index_train);
% Test Selection and mixing
X_test=X(Index_test,:);
Labels_test=Labels(Index_test);
Index_test=randperm(length(Labels_test));
X_test=X_test(Index_test,:);
Labels_test=Labels_test(Index_test);
clear Index_train Index_test index i_class N_i_class I_train I_test
%% OPCION dibujos de imagenes 
if i_dib==1
    figure('name','Microarray por clases')
    for i_class=1:length(V_clases);
        index=find(Labels==V_clases(i_class));
        subplot(4,2,i_class)
        hold on
        for i_aux=1:length(index);
            plot(X(index(i_aux),:));
        end
        ylabel(V_clases(i_class))
    end   
    figure('name','Microarray Medidas')
    imagesc(X);
    colorbar
    xlabel('coordenada')
    ylabel('muestra')
end
clear i_dib index i_aux i_class

%% Create a knn classifier:
knnclass = fitcknn(X_train,Labels_train,'NumNeighbors',K_neig);
knn_out = predict(knnclass,X_train);
knn_Pe_train=sum(Labels_train ~= knn_out)/length(Labels_train);
fprintf(1,' error knn train = %g   \n', knn_Pe_train)
% Train confusion matrix
CM_knn_train=confusionmat(Labels_train,knn_out)
knn_out = predict(knnclass,X_test);
knn_Pe_test=sum(Labels_test ~= knn_out)/length(Labels_test);
fprintf(1,' error knn test = %g   \n', knn_Pe_test)
% Test confusion matrix
CM_knn_test=confusionmat(Labels_test,knn_out)