# Rede Neural para Segmentar Frutos de Café

Este estudo tem como base a abordagem de aprendizado de máquina para redes neurais convolucionais (RNC), mais especificamente da Rede Neural U-Net. Uma rede neural convolucional é capaz de aplicar filtros em dados visuais, mantendo a relação de vizinhança entre os pixels da imagem ao longo do processamento da rede. Desta maneira as RNCs, incluída a U-Net, utiliza de técnicas de aprendizado profundo para, a partir da uma imagem de entrada, identificar e diferenciar os dados para construir um produto definida pelo código. No caso do presente trabalho a entrada é uma imagem normal de um pé de café e o produto é uma máscara que diz os locais onde possuem frutos de café. 

Os modelos de RNC de aprendizado profundo são usados para treinar algoritmos com diversas imagens com saídas conhecidas e testá-los posteriormente com imagens com produtos desconhecidos. Seguindo a ótica deste estudo, a rede será treinada com entradas figuras de café e uma máscara pronta, previamente definida. Assim, seu produto será uma rede capaz de desenvolver automaticamente uma máscara para uma imagem nova. 

O modelo foi desenvolvido com uma amostra de doze imagens de cafés coletados de uma base construída por cinco especialistas de café e agronomia da Universidade Federal de Uberlândia (UFU). Estas imagens foram obtidas de frutos de café de diferentes regiões do Brasil de forma colaborativa com produtores, pesquisadores e estudantes. Manteve-se um padrão para captura de imagens onde o usuário define a posição do dossel (local onde é localizado os frutos do café), abre a copa da planta na posição e aproxima o celular na posição vertical ou horizontal. 

A amostra levou em consideração selecionar imagens que levando em consideração diferentes casos como a quantidade e estágio de frutos, interferência da luz na imagem e presença de solo ou céu. Após a etapa de coleta foram criadas a máscaras, com auxílio de programas de edição, para os doze cafés selecionados. Na máscara, a localização dos frutos do café é colocada de branco em contraste ao fundo preto da figura. 

Sob esta ótica, a máscara é uma imagem bitmap, ou seja, uma matriz de zeros (preto) e uns (branco). Como foi discutido anteriormente, o modelo U-Net precisa ser treinado com imagens de treinamento e validação, das doze selecionadas, foram tomadas aleatoriamente dez imagens para treinamento e duas para validação. 

Foram tomadas 20% das imagens (duas) aleatoriamente para a camada de validação e 80% (dez) para o treino. Após a organização da base de dados os parâmetros de treinos foram ajustados diversas vezes, até que, empiricamente, o melhor resultado foi: 

    Entrada de Arquivos em 128x128 

    Função de ativação sigmoide para a saída, para torná-la binária 

    Saída das imagens com uma camada (preto ou branco) ao invés de tons de cinza (duas camadas) 

    Valor limiar de 1,8 para selecionar os bits de 1 e 0 da saída 

    10 Épocas de treino, 5 passos por época e tamanho de batch 2 

    Otimizador Adam

Após a construção do modelo U-Net foram selecionadas cinco figuras aleatórias do banco de imagens de café previamente citado. Estas figuras foram utilizadas para testar a rede para analisar a máscara criada.

Mediante essa perspectiva, ao analisar o produto dos testes finais foi constatado que a rede tem dificuldade com figuras que possuem cafés verdes, por conta dos mesmos tons das folhas e frutos, e quando possui um céu luminoso na imagem a rede o classifica como café. Isso pode ser respondido pela falta de imagens, visto que a rede foi treinada somente com doze, assim, não passou por diversas maneiras que uma imagem pode chegar para a rede. 

Entretanto, as imagens com cafés cereja, passa e seco foram classificadas com um acerto considerável, somente com erros nos casos citados acima. 
