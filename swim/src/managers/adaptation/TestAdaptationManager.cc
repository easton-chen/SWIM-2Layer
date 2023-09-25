/*******************************************************************************
 * Simulator of Web Infrastructure and Management
 * Copyright (c) 2016 Carnegie Mellon University.
 * All Rights Reserved.
 *  
 * THIS SOFTWARE IS PROVIDED "AS IS," WITH NO WARRANTIES WHATSOEVER. CARNEGIE
 * MELLON UNIVERSITY EXPRESSLY DISCLAIMS TO THE FULLEST EXTENT PERMITTED BY LAW
 * ALL EXPRESS, IMPLIED, AND STATUTORY WARRANTIES, INCLUDING, WITHOUT
 * LIMITATION, THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE, AND NON-INFRINGEMENT OF PROPRIETARY RIGHTS.
 *  
 * Released under a BSD license, please see license.txt for full terms.
 * DM-0003883
 *******************************************************************************/

#include "TestAdaptationManager.h"
#include "managers/adaptation/UtilityScorer.h"
#include "managers/execution/AllTactics.h"
#include <fstream>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <unistd.h>

using namespace std;
using namespace omnetpp;

Define_Module(TestAdaptationManager);

/**
 * Reactive adaptation
 *
 * RT = response time
 * RTT = response time threshold
 *
 * - if RT > RTT, add a server if possible, if not decrease dimmer if possible
 * - if RT < RTT and spare utilization > 1
 *      -if dimmer < 1, increase dimmer else if servers > 1 and no server booting remove server
 */
Tactic* TestAdaptationManager::evaluate() {
    MacroTactic* pMacroTactic = new MacroTactic;
    Model* pModel = getModel();
    const double dimmerStep = 1.0 / (pModel->getNumberOfDimmerLevels() - 1);
    double dimmer = pModel->getDimmerFactor();
    double spareUtilization =  pModel->getConfiguration().getActiveServers() - pModel->getObservations().utilization;
    bool isServerBooting = pModel->getServers() > pModel->getActiveServers();
    double responseTime = pModel->getObservations().avgResponseTime;
    double timeoutRate = pModel->getObservations().timeoutRate;
    double avgInterval = pModel->getEnvironment().getArrivalMean();
    double avgThroughput = 1.0 / avgInterval;

    int socket_desc;
	struct sockaddr_in server;
	char message[100];
    char server_reply[100];
	
	// 创建socket
	socket_desc = socket(AF_INET, SOCK_STREAM, 0);
	if (-1 == socket_desc) {
		perror("connot create socket");
		exit(1);
	}
	
	server.sin_addr.s_addr = inet_addr("127.0.0.1");
	server.sin_family = AF_INET;
	server.sin_port = htons(50007);
	
	// connect to controller
	if (connect(socket_desc, (struct sockaddr*)&server, sizeof(server)) < 0) {
		perror("SWIM connot connect to controller!");
	}

    // send data

    sprintf(message, "%lf", timeoutRate);
    if (::send(socket_desc, message, strlen(message), 0) < 0) {
        perror("SWIM send data error");
    }
    
    printf("SWIM send message 'y = %lf' success\n", timeoutRate);
    // receive data
    if (recv(socket_desc, server_reply, 100, 0) < 0) {
        perror("SWIM recv error");
    }
    
    printf("SWIM recv success: \n");
    //puts(server_reply);
	
	close(socket_desc);

    const char s[2] = " ";
    char *token;
    
    // dimmer
    token = strtok(server_reply, s);
    dimmer = atof(token);
    printf( "dimmer = %lf\n", dimmer);
    // server
    token = strtok(NULL, s); 
    int serverNum = atol(token); 
    /*if(avgThroughput < 20){
        serverNum = 1;
    }*/
    printf( "server num = %d\n", serverNum);


    pMacroTactic->addTactic(new SetDimmerTactic(dimmer));
    int dServer = serverNum - pModel->getServers();
    if(dServer > 0){
        while(dServer--){
            if(!isServerBooting){
                pMacroTactic->addTactic(new AddServerTactic);
            }
        }
    } else if(dServer < 0){
        while(dServer++){
            if(!isServerBooting){
                pMacroTactic->addTactic(new RemoveServerTactic);
            }
        }
    }
    
    return pMacroTactic;
}
