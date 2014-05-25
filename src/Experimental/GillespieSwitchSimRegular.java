package Experimental;

/**
 * @(#)  GillespieSwitchSim
 */

import GPUBackend.OpenCLHandler;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.LinkedList;
import java.util.Random;


/**
 *      GillespieSwitchSim 
 *
 *   <br>
 * 
 * @author James B. Silva <jbsilva@bu.edu>
 * @date May 2012
 */
public class GillespieSwitchSimRegular {

    private OpenCLHandler clHandler;
    private int NumOfSystems = 256;
    private int p1initial = 40;
    private int p2initial = 0;
    private float beta = 9.0f;
    private float alpha1  = beta*10;
    private float alpha2 = beta*10;
    private float tau1 = beta*1.0f;
    private float tau2 = beta*1.0f;
    private float kappa1 = beta*1.0f;
    private float kappa2 = beta*1.0f;
    private float n1 = 2.0f;
    private float n2 = 2.0f;
    private int MCsteps = 0;
    private int CheckMeasureTime = 100;
    private int MaxSteps = 10000000;
    private String dataDirectory = "/home/j2/Classwork/Data/";
    private String dataFilename =  "gillespieSwitchRegData.txt";
    private boolean findAverage =true;
    private boolean takeData=false;
    private boolean showRates=false;
    private Random ran;
    private int[] p1;
    private int[] p2;
    private float[] time;
    private float[] tMeasured;
    private float[] SysParameters;
    private boolean profileTime = false;
    
    public void initialize(){
    
        float[] sysparams = new float[9];
        sysparams[0] = alpha1;
        sysparams[1] = alpha2;
        sysparams[2] = tau1;
        sysparams[3] = tau2;
        sysparams[4] = n1;
        sysparams[5] = n2;
        sysparams[6] = kappa1;
        sysparams[7] = kappa2;
        sysparams[8] = (float) p1initial;
        
        SysParameters = sysparams;
    
        p1 = new int[NumOfSystems];
        p2 = new int[NumOfSystems];
        time = new float[NumOfSystems];
        tMeasured = new float[NumOfSystems];
        
        for (int i=0;i<NumOfSystems;i++){
            p1[i]=p1initial;
            p2[i]=p2initial;
            time[i] = 0.0f;
            tMeasured[i] = 0.0f;
        }
        

        ran = new Random();
        
    }
    
    public void doOneStep(){
        MCsteps++;
        doGillespieStep(SysParameters);
        
        // testing r values
        if(showRates){
        printRvalues(p1[0], p2[0]);}
        
        if((MCsteps % 10000)==0){
            System.out.println("MC Step: "+MCsteps);
        }
        if((MCsteps % CheckMeasureTime)==0){
            checkMeasurements();
            // clear data
            for(int i =0;i < NumOfSystems;i++){
                tMeasured[i]=0.0f;
            }
        }
    }
    
    public void RunSimulation(){
        double sum=0;
        for(int i = 0; i < MaxSteps;i++){
            long time = System.nanoTime();
            doOneStep();
            if(profileTime){time = System.nanoTime()-time;
            sum = time/1000000+sum;
            System.out.println("Time for step: "+(time/1000000)+" ms");
            if((i%10)==0){
            System.out.println("AVG Time for step: "+(sum/10.0)+" ms");
            sum =0;
            }
            }
        }
    
    }
    
    private void doGillespieStep(float[] sysParam){
        
    for(int currSys=0;currSys<this.NumOfSystems;currSys++){    
    // get parameters
    int nparams = 9;
    float alpha1 =  sysParam[0];
    float alpha2 = sysParam[1];
    float tau1 = sysParam[2];
    float tau2 = sysParam[3];
    float n1 = sysParam[4];
    float n2 = sysParam[5];
    float kappa1 = sysParam[6];
    float kappa2 = sysParam[7];

    // Calculate R ? Does this happen before time update
    float rtemp = 0.0f;
    
        // birth
        rtemp = (float) (rtemp+ alpha1/(1+Math.pow(p2[currSys]/kappa2,n1))); 
        rtemp = (float) (rtemp+ alpha2/(1+Math.pow(p1[currSys]/kappa1,n2)));
    
        // death 
        rtemp = rtemp+ p1[currSys]/tau1; 
        rtemp = rtemp+ p2[currSys]/tau2;
    
        float rCurr = rtemp;

        float r1 = (float) ((alpha1/(1+Math.pow(p2[currSys]/kappa2,n1)))/rCurr);
        float r2 = (float) (r1+((alpha2/(1+Math.pow(p1[currSys]/kappa1,n2)))/rCurr));
        float r3 = r2+((p1[currSys]/tau1)/rCurr);

    // update time
    time[currSys] = (float) (time[currSys]-1.0*Math.log(ran.nextFloat())/rCurr);

    float ran2 =ran.nextFloat();
    // Determine process to update
        if(ran2 < r1){
            p1[currSys] = p1[currSys]+1;
        }else if(r1 < ran2  && ran2 < r2){
            p2[currSys] = p2[currSys]+1;
        }else if(r2 < ran2  && ran2 < r3){
            p1[currSys] = p1[currSys]-1;
        }else{
            p2[currSys] = p2[currSys]-1;
        }


    // Determine if transitioned
        if(p2[currSys] > p1initial){
            tMeasured[currSys] = time[currSys];
            p1[currSys]=p1initial;
            p2[currSys]=p2initial;
            time[currSys]=0.0f;
            }
        }
    }
    
    public void saveData(LinkedList<Double> data){
                int size = data.size();
        try {
            PrintStream out = new PrintStream(new FileOutputStream(
                dataDirectory+dataFilename,true));
            for (int i = 0; i < (size); i++){
                    out.println(data.remove());
            }
            out.println();
            out.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }
    }
    
    
    
    private void checkMeasurements(){
        if(findAverage){
            System.out.println("Average p1 : "+averageP1(1000));
            System.out.println("Average p2 : "+averageP2(1000));
        
        }
        
        if(takeData){
        LinkedList<Double> data = new LinkedList<Double>(); 
        for(int i = 0;i<tMeasured.length;i++){
            if(tMeasured[i]>0){
                data.push(((double)tMeasured[i]));
            }
        }
        saveData(data);}
    }
    
    public double averageP1(int navg){
        LinkedList<Double> data = new LinkedList<Double>(); 
        long sum =0;
        for(int i = 0;i< p1.length;i++){
            sum = sum + p1[i];
        } 
        double avg = ((double)sum)/((double) p1.length);
        return avg;
    }
    
    
    public double averageP2(int navg){
        LinkedList<Double> data = new LinkedList<Double>(); 
        long sum =0;
        for(int i = 0;i< p2.length;i++){
            sum = sum + p2[i];
        } 
        double avg = ((double)sum)/((double) p2.length);
        return avg;
    }
    
    public void printRvalues(int p1, int p2){
    
       // Calculate R ? Does this happen before time update
       float rtemp = 0.0f;
    
        // birth
        rtemp = (float) (rtemp+ (alpha1/(1+Math.pow(p2/kappa2,n1)))); 
        rtemp = (float) (rtemp+ (alpha2/(1+Math.pow(p1/kappa1,n2))));
    
        // death 
        rtemp = rtemp+ (p1/tau1); 
        rtemp = rtemp+ (p2/tau2);
    
        float rCurr = rtemp;

        float r1 = (float) ((alpha1/(1+Math.pow(p2/kappa2,n1)))/rCurr);
        float r2 = (float) (((alpha2/(1+Math.pow(p1/kappa1,n2)))/rCurr));
        float r3 = ((p1/tau1)/rCurr);
        float r4 = ((p2/tau2)/rCurr);
        
        System.out.println("****************");
        System.out.println("p1: "+p1);
        System.out.println("p2: "+p2);
        
        System.out.println("r1: "+r1);
        System.out.println("r2: "+r2);
        System.out.println("r3: "+r3);
        System.out.println("r4: "+r4);
        System.out.println("****************");
    }
    
    
    public static void main(String[] args) throws IOException {
        GillespieSwitchSimRegular sim = new GillespieSwitchSimRegular();
        sim.initialize();
        sim.RunSimulation();
    }
}
