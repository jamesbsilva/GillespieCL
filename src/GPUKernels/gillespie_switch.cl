__kernel void gillespie_switch(__global int *p1,__global int *p2,__global float *time,
            __global const float *sysParam,__global float * ran,__global float *tMeasured, int nElements) {

    // Get the index of the current element to be processed
    int currSys = get_global_id(0);

    // bound check, equivalent to the limit on a 'for' loop
    if (currSys >= nElements)  {
        return;
    }

    // get parameters
    int nparams = 10;
    float alpha1 =  sysParam[currSys*nparams+0];
    float alpha2 = sysParam[currSys*nparams+1];
    float tau1 = sysParam[currSys*nparams+2];
    float tau2 = sysParam[currSys*nparams+3];
    float n1 = sysParam[currSys*nparams+4];
    float n2 = sysParam[currSys*nparams+5];
    float kappa1 = sysParam[currSys*nparams+6];
    float kappa2 = sysParam[currSys*nparams+7];
    int p1init = (int) sysParam[currSys*nparams+8];
    int p2init = (int) sysParam[currSys*nparams+9];

    // Calculate R ? Does this happen before time update
    float rtemp = 0.0f;
    
        // birth
        rtemp = rtemp+ alpha1/(1+pow(p2[currSys]/kappa2,n1)); 
        rtemp = rtemp+ alpha2/(1+pow(p1[currSys]/kappa1,n2));
    
        // death 
        rtemp = rtemp+ p1[currSys]/tau1; 
        rtemp = rtemp+ p2[currSys]/tau2;
    
        float rCurr = rtemp;

        float r1 = (alpha1/(1+pow(p2[currSys]/kappa2,n1)))/rCurr;
        float r2 = r1+(alpha2/(1+pow(p1[currSys]/kappa1,n2)))/rCurr;
        float r3 = r2+(p1[currSys]/tau1)/rCurr;

    // update time
    time[currSys] = time[currSys]-1.0*log(ran[currSys*2])/rCurr;
    float currTime = time[currSys];

    // Determine process to update
        if(ran[currSys*2+1] < r1){
            p1[currSys] = p1[currSys]+1;
        }else if(r1 < ran[currSys*2+1]  && ran[currSys*2+1] < r2){
            p2[currSys] = p2[currSys]+1;
        }else if(r2 < ran[currSys*2+1]  && ran[currSys*2+1] < r3){
            p1[currSys] = p1[currSys]-1;
        }else{
            p2[currSys] = p2[currSys]-1;
        }


    // Determine if transitioned. reset system if so
    if(p2[currSys] > p1init){
        tMeasured[currSys] = currTime;
        p1[currSys] = p1init;
        p2[currSys] = p2init;
        time[currSys] = 0.0f;
    }
}
