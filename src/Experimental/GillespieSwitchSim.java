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
public class GillespieSwitchSim {

    private OpenCLHandler clHandler;
    private int NumOfSystems = 128000;
    private int p1initial = 160;
    private int p2initial = 0;
    private float beta = 4.0f;
    private float alpha1 = beta*10;
    private float alpha2 = beta*10;
    private float tau1 = beta*1.0f;
    private float tau2 = beta*1.0f;
    private float kappa1 = beta*1.0f;
    private float kappa2 = beta*1.0f;
    private float n1 = 2.0f;
    private float n2 = 2.0f;
    private int GlobalWorkSize = NumOfSystems;
    private int LocalWorkSize = 64;
    private String gillespieKernelName = "gillespie_switch";
    private String RNGKernelName = "mersenne_twister";   
    private int MCsteps = 0;
    private int CheckMeasureTime = 5000;
    private String fillFlKernelName = "fill_float_buffer";
    private int MaxSteps = 100000000;
    private String dataDirectory = "/home/j2/Classwork/Data/";
    private String dataFilename =  "gillespieSwitchData40.txt";
    private boolean findAverage =false;
    private boolean takeData=true;
    private String DeviceType="CPU";
    private Random ran;
    private boolean profileTime = false;
    
    public void initialize(){
    
        float[] sysparams = new float[10];
        sysparams[0] = alpha1;
        sysparams[1] = alpha2;
        sysparams[2] = tau1;
        sysparams[3] = tau2;
        sysparams[4] = n1;
        sysparams[5] = n2;
        sysparams[6] = kappa1;
        sysparams[7] = kappa2;
        sysparams[8] = (float) p1initial;
        sysparams[9] = (float) p2initial;
    
        
        ran = new Random();
        
        clHandler = new OpenCLHandler();
        
        clHandler.initializeOpenCL(DeviceType);
        
        clHandler.createKernel("", gillespieKernelName);
        clHandler.createKernel("", RNGKernelName);
        clHandler.createKernel("", fillFlKernelName);

        // random numbers
        clHandler.createFloatBuffer(RNGKernelName, 0, NumOfSystems*2, 0.0f, 0, true);
        //seeds
        clHandler.createIntArg(RNGKernelName, 0, ran.nextInt());
        clHandler.createIntArg(RNGKernelName, 1, ran.nextInt());
        clHandler.createIntArg(RNGKernelName, 2, 2);

        
        // initialize protein amounts
        clHandler.createIntBuffer(gillespieKernelName, 0, NumOfSystems, p1initial, 0, true);
        clHandler.createIntBuffer(gillespieKernelName, 1, NumOfSystems, p2initial, 0, true);
        // time
        clHandler.createFloatBuffer(gillespieKernelName, 0, NumOfSystems, 0.0f, 0, true);
        // parameters
        clHandler.createFloatBuffer(gillespieKernelName, 1, NumOfSystems*sysparams.length, sysparams, 2, false,true);
        // random numbers
        clHandler.copyFlBufferAcrossKernel(RNGKernelName, 0,gillespieKernelName , 2);

        // measured switch times
        clHandler.createFloatBuffer(gillespieKernelName, 3, NumOfSystems, 0.0f, 0, true);
        
        // num of Elements
        clHandler.createIntArg(gillespieKernelName,0, NumOfSystems);
        // num of Elements
        //clHandler.createIntArg(RNGKernelName,1, NumOfSystems*2);
       
        
        System.out.println("Using Device MB : "+clHandler.getDeviceUsedMB(gillespieKernelName));
        
        // third kernel
        clHandler.copyFlBufferAcrossKernel(gillespieKernelName, 3, fillFlKernelName, 0);
        clHandler.createFloatArg(fillFlKernelName, 0, 0.0f);
        
        clHandler.setKernelArg(gillespieKernelName);
        clHandler.setKernelArg(RNGKernelName);
        clHandler.setKernelArg(fillFlKernelName);
        
    }
    
    public void doOneStep(){
        MCsteps++;
        
        // set seeds
        clHandler.createIntArg(RNGKernelName, 0, Math.abs(2*ran.nextInt()));
        clHandler.createIntArg(RNGKernelName, 1, Math.abs(2*ran.nextInt()));
        clHandler.setKernelArg(RNGKernelName, true); 
        clHandler.runKernel(RNGKernelName,GlobalWorkSize,LocalWorkSize);
        clHandler.getFloatBufferAsArray(RNGKernelName, 0, 1, false);
 
        clHandler.getFloatBufferAsArray(gillespieKernelName, 0, 1, false);
        
        /*if((MCsteps % 10)==0){
            System.out.println("-----------------------------------");
            System.out.print("Random In Mersenne: ");
            clHandler.getFloatBufferAsArray(RNGKernelName, 0, 2, false);
 
            System.out.println("Random In Gillespie: ");
            System.out.println("___________________");
            clHandler.getFloatBufferAsArray(gillespieKernelName, 2, 2, true);
            System.out.println("___________________");
            System.out.println("time: ");
            clHandler.getFloatBufferAsArray(gillespieKernelName, 0, 1, true);
            
            //System.out.println("Mersenne Seed 1: "+clHandler.getIntArg(RNGKernelName, 0));
            //System.out.println("Mersenne Seed 2: "+clHandler.getIntArg(RNGKernelName, 1));
        }*/
        
        
        clHandler.setKernelArg(gillespieKernelName, true);
        clHandler.runKernel(gillespieKernelName,GlobalWorkSize,LocalWorkSize);
        clHandler.getIntBufferAsArray(gillespieKernelName, 0, 1, false);
        clHandler.getIntBufferAsArray(gillespieKernelName, 1, 1, false);
        clHandler.getFloatBufferAsArray(gillespieKernelName, 0, 1, false);
        clHandler.getFloatBufferAsArray(gillespieKernelName, 3, 1, false);
        
        /*
        int[] push;int p1;int p2;
        if((MCsteps % 10)==0){
            System.out.print("p1: ");
            push = clHandler.getIntBufferAsArray(gillespieKernelName, 0, 1, true);
            p1=push[0];
            System.out.print("p2: ");
            push = clHandler.getIntBufferAsArray(gillespieKernelName, 1, 1, true);
            p2=push[0];
            printRvalues(p1, p2);
            System.out.print("time: ");
            clHandler.getFloatBufferAsArray(gillespieKernelName, 0, 1, true);
            System.out.print("tTransition: ");
            clHandler.getFloatBufferAsArray(gillespieKernelName, 3, 1, true);
        }*/
        
        if((MCsteps % 1000)==0){
            System.out.println("MC Step: "+MCsteps);
        }
        if((MCsteps % CheckMeasureTime)==0){
            checkMeasurements();
            // clear data
            clHandler.getFloatBufferAsArray(gillespieKernelName, 3, 1, false);
            clHandler.getFloatBufferAsArray(fillFlKernelName, 0, 1, false);
            clHandler.setKernelArg(fillFlKernelName, true);
            clHandler.runKernel(fillFlKernelName,GlobalWorkSize,LocalWorkSize);
            clHandler.getFloatBufferAsArray(fillFlKernelName, 0, 1, false);
            clHandler.getFloatBufferAsArray(gillespieKernelName, 3, 1, false);
        }
    }
    
    public void RunSimulation(){
        
        double sum = 0;
        for(int i = 0; i < MaxSteps;i++){
            long time = System.nanoTime();
            doOneStep();
            if(profileTime){time = System.nanoTime()-time;
            sum =sum+(time/1000000);
            System.out.println("Time for step: "+(time/1000000)+" ms");
                if((i%10)==0){
            System.out.println("AVG Time for step: "+(sum/10.0)+" ms");
            sum =0;
            }
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
    
    private int[] getSeedSystemsArray(){
        int[] seeds = new int[NumOfSystems*2];
        for(int i=0;i<seeds.length;i++){
        seeds[i] = (int)(274252423*Math.random());}
        return seeds;
    }
    private void checkMeasurements(){
        if(findAverage){
            System.out.println("Average p1 : "+averageP1(1000));
        }
        
        if(takeData){
        //System.out.println("Get buffer.");
        float[] switchTimes = clHandler.getFloatBufferAsArray(gillespieKernelName, 3, NumOfSystems,false);
        LinkedList<Double> data = new LinkedList<Double>(); 
        //System.out.println("Obtaining data.");        
        for(int i = 0;i<switchTimes.length;i++){
            if(switchTimes[i]>0.0){data.push(((double)switchTimes[i]));}
        }
        //System.out.println("Saving.");
        saveData(data);}
    }
    
    public double averageP1(int navg){
        int[] p1 = clHandler.getIntBufferAsArray(gillespieKernelName, 0, navg,false);
        LinkedList<Double> data = new LinkedList<Double>(); 
        long sum =0;
        for(int i = 0;i< p1.length;i++){
            sum = sum + p1[i];
        } 
        double avg = ((double)sum)/((double) p1.length);
        return avg;
    }
    
    
    public void printRvalues(int p1, int p2){
    
       // Calculate R ? Does this happen before time update
       float rtemp = 0.0f;
    
        // birth
        rtemp = (float) (rtemp+ alpha1/(1+Math.pow(p2/kappa2,n1))); 
        rtemp = (float) (rtemp+ alpha2/(1+Math.pow(p1/kappa1,n2)));
    
        // death 
        rtemp = rtemp+ p1/tau1; 
        rtemp = rtemp+ p2/tau2;
    
        float rCurr = rtemp;

        float r1 = (float) ((alpha1/(1+Math.pow(p2/kappa2,n1)))/rCurr);
        float r2 = (float) ((alpha2/(1+Math.pow(p1/kappa1,n2)))/rCurr);
        float r3 = (p1/tau1)/rCurr;
        float r4 = (p2/tau2)/rCurr;
        
        System.out.println("****************");
        System.out.println("r1: "+r1);
        System.out.println("r2: "+r2);
        System.out.println("r3: "+r3);
        System.out.println("r4: "+r4);
        System.out.println("****************");
    }
    
    
    public static void main(String[] args) throws IOException {
        GillespieSwitchSim sim = new GillespieSwitchSim();
        sim.initialize();
        sim.RunSimulation();
    }
}
