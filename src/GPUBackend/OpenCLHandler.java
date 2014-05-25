package GPUBackend;
/*
 *   @(#)   OpenCLHandler
 */

import com.jogamp.opencl.CLDevice.Type;
import static com.jogamp.opencl.CLMemory.Mem.*;
import com.jogamp.opencl.*;
import com.jogamp.opencl.util.CLPlatformFilters;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import static java.lang.System.out;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.HashMap;

/**
*      OpenCLHandler handles OpenCL devices by initializing OpenCL and creating
*   buffers/kernels.
* 
* <br>
* 
* @author      James B. Silva <jbsilva @ bu.edu>                 
* @since       2012-05
*/
public class OpenCLHandler {
    private CLPlatform[] clPl;
    private CLContext context;
    private CLDevice device;
    private CLCommandQueue queue;
    private HashMap<String,CLKernel> kernels;
    private HashMap<String,ArrayList<CLBuffer<FloatBuffer>>> flBuffers;
    private HashMap<String,ArrayList<CLBuffer<IntBuffer>>> intBuffers;
    private HashMap<String,ArrayList<String>> argTypes;
    private HashMap<String,ArrayList<Integer>> intArgs;
    private HashMap<String,ArrayList<Float>> floatArgs;    
    private HashMap<String,ArrayList<Long>> longArgs;
    
    /**
    *   initializeOpenCL setups the OpenCL context to run the simulation in the device.
    * 
    * @param deviceType - string for specific device. "GPU","CPU","NVIDIA" example supported types
    *   "" for no preference.
    */
    public void initializeOpenCL(String deviceType){
    
        // Initialize all list and maps
        kernels = new HashMap<String,CLKernel>();
        flBuffers = new HashMap<String,ArrayList<CLBuffer<FloatBuffer>>> ();
        intBuffers = new HashMap<String,ArrayList<CLBuffer<IntBuffer>>>();
        argTypes= new HashMap<String,ArrayList<String>>();
        intArgs = new HashMap<String,ArrayList<Integer>>();
        floatArgs  = new HashMap<String,ArrayList<Float>>();
        longArgs = new HashMap<String,ArrayList<Long>>();
        
        // search for platform support given device string
        if(deviceType.equalsIgnoreCase("GPU") || 
            deviceType.equalsIgnoreCase("Graphics Processor") ||
            deviceType.equalsIgnoreCase("Graphics")){
            clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(Type.GPU));}
        else if(deviceType.equalsIgnoreCase("CPU") || 
            deviceType.equalsIgnoreCase("Processor")){
            clPl = CLPlatform.listCLPlatforms(CLPlatformFilters.type(Type.CPU));
        }else{
            clPl = CLPlatform.listCLPlatforms();
        }
        
        if(deviceType.equalsIgnoreCase("NVIDIA")){
            deviceType = "NVIDIA CUDA";
        }
        
        if(deviceType.equalsIgnoreCase("")){
            System.out.println("List Platforms");
        }else{
            System.out.println("Searching Platforms For Device : " +deviceType);
        }
        
        int deviceid = 0;
        // If looking for vendor specific type look for it otherwise list all devices
        for(int i =0;i < clPl.length;i++){
            if(clPl[i].getName().equalsIgnoreCase(deviceType) || 
                    deviceType.equals("") || deviceType.equals("GPU") 
                    || deviceType.equals("CPU")){
            deviceid = i;
            System.out.println(clPl[i].getName());
            System.out.println("Devices per platform :"+clPl[i].listCLDevices().length);
            System.out.println("Compute Devices on max flops device:"+
                clPl[i].getMaxFlopsDevice().getMaxComputeUnits());
            System.out.println(" Max flops device:"+
                clPl[i].getMaxFlopsDevice().getMaxClockFrequency());
            System.out.println(" Max global memory for device:"+
                (clPl[i].getMaxFlopsDevice().getGlobalMemSize()/1000000));
            
            System.out.println(" Max memory allocateable size for device:"+
                (clPl[i].getMaxFlopsDevice().getMaxMemAllocSize()/1000000));
             
            }        
        }
    
        // set up (uses default CLPlatform and creates context for all devices)
        context = CLContext.create(clPl[deviceid]);
        out.println("created "+context);
    
        // select fastest device
        device = context.getMaxFlopsDevice();
        out.println("using "+device);

        // create command queue on device.
        queue = device.createCommandQueue();
    }
    
    /**
    *   createKernel builds the kernel in the OpenCL device.
    * 
    * @param fname - kernel file name- null for device filename is the same 
    *       as kernelname but cl as filetype
    * @param kernelname - kernel name 
    */
    public void createKernel(String fname,String kernelname){
            
        // Find kernel file
        if(fname==null || fname.equals("")){fname= kernelname+".cl";}
        String fname1 = "GPUKernels/"+fname;
        File f = new File(fname1);

        System.out.println("Current Path:"+f.getAbsolutePath());
        if(f.exists()){
            System.out.println("Found Kernel File");
        }else{
            System.out.println("Assuming running from IDE folder.");
            String fname2 = "src/"+fname1;
            f = new File(fname2);
            if(f.exists()){
                System.out.println("Found Kernel File");
                fname = fname2;
            }else{
                System.out.println("Kernel File Not Found");}
        }

        System.out.println("Creating kernel:"+kernelname);

        // Read source code
        CLProgram program=null;

        try{
        String sourceCode = readFile(fname);

        argTypes.put(kernelname,getKernelIOTypes(kernelname,sourceCode));

        // load sources, create and build program
        program = context.createProgram(sourceCode).build();

        System.out.println("prog: "+program);
        }catch (Exception e) {
            e.printStackTrace();
        }       

        // kernel comes labeled therefore just push into kernels
        kernels.put(kernelname,program.createCLKernel(kernelname));
          
    }
    
    /**
    *       runKernel runs the kernel in the OpenCL device.
    * 
    * @param kernelname - kernel name of the kernel to be run
    * @param gsize - global work size
    * @param lsize - local work size
    */
    public void runKernel(String kernelname, int gsize,int lsize){
        // Assert kernel buffers are setup 
        assertKernelBuffersMade(kernelname);
        
        setQueueBuff(kernelname);
        queue.put1DRangeKernel(kernels.get(kernelname), 0, gsize, lsize);            
            
    }
    
    /**
    *       setQueueBuff sets the buffers in the OpenCL queue so the device knows  
    *   which buffers to access.
    * 
    * @param kernelname - kernel which needs buffers in queue
    */
    private void setQueueBuff(String kernelname){
        // get list of buffers for this kernel
        ArrayList<String> kernelTypes = argTypes.get(kernelname);
        int floatBuffInd = 0;
        int intBuffInd = 0;
        String typecurr;
        
        // Set as buffers for the queue
        for(int j =0;j<kernelTypes.size();j++){
            typecurr = kernelTypes.get(j);
            if(typecurr.contains("int")&& typecurr.contains("buffer")){
                ArrayList<CLBuffer<IntBuffer>> outBuffers = intBuffers.get(kernelname);
                CLBuffer<IntBuffer> push = outBuffers.get(intBuffInd);
                queue.putWriteBuffer(push,false);
                intBuffInd++;
            }else if(typecurr.contains("float") && typecurr.contains("buffer")){
                //System.out.println("Queuing Float Buffer");
                ArrayList<CLBuffer<FloatBuffer>> outBuffers = flBuffers.get(kernelname);
                CLBuffer<FloatBuffer> push = outBuffers.get(floatBuffInd);
                queue.putWriteBuffer(push,false);
                floatBuffInd++;
            }
        }
    }
   
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial value of all buffer entries
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    */
    public void createIntBuffer(String kernelname,int argn,int size, int s0,int readwrite,boolean fill){
        // Create clbuffer object
        CLBuffer<IntBuffer> buffer;
        if(readwrite ==2){
            buffer = context.createIntBuffer(size, WRITE_ONLY);
        }else if(readwrite==1){
            buffer = context.createIntBuffer(size, READ_ONLY);
        }else{
            buffer = context.createIntBuffer(size, READ_WRITE);
        }    
        
        // fill the buffer if required
        if(fill){
        buffer.use(fillIntBuffer(buffer.getBuffer(),s0));}
 
        // Initialize buffer type array if not already
        ArrayList<CLBuffer<IntBuffer>> kbuff = this.intBuffers.get(kernelname);
                if(kbuff == null){ 
            System.err.println("Initializing int kernel buffers with value "+s0+" .");
            kbuff = new  ArrayList<CLBuffer<IntBuffer>>();}

        // push buffer into buffers list        
        kbuff.add(argn, buffer);
        intBuffers.put(kernelname, kbuff);    
    }
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    */
    public void createIntBuffer(String kernelname,int argn,int size, int[] s0,int readwrite, boolean fill){
        createIntBuffer(kernelname,argn,size,s0,readwrite,fill, false);
    }
    
    /**
    *       createIntBuffer creates and fills an integer buffer then adds it to 
    *   the list of int buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    * @param modfill - fill using a modulus fill which fills based on mod of array size
    */
    public void createIntBuffer(String kernelname,int argn,int size, int[] s0,int readwrite, boolean fill, boolean modFill){
        // Create clbuffer object
        CLBuffer<IntBuffer> buffer;
        if(readwrite ==2){
            buffer = context.createIntBuffer(size, WRITE_ONLY);
        }else if(readwrite==1){
            buffer = context.createIntBuffer(size, READ_ONLY);
        }else{
            buffer = context.createIntBuffer(size, READ_WRITE);
        }    
        
        // fill the buffer if required
        if(fill){
        buffer.use(fillIntBuffer(buffer.getBuffer(),s0));}else if(modFill){
            buffer.use(fillIntBufferMod(buffer.getBuffer(),s0));
        }
        
        // Initialize buffer type array if not already
        ArrayList<CLBuffer<IntBuffer>> kbuff = this.intBuffers.get(kernelname);
            if(kbuff == null){ 
            System.err.println("Initializing int kernel buffers.");
            kbuff = new  ArrayList<CLBuffer<IntBuffer>>();}

        // push buffer into buffers list               
        kbuff.add(argn, buffer);
        intBuffers.put(kernelname, kbuff);
    }
    
    
    
    /**
    *       fillIntBuffer fills an int buffer with given value.
    * 
    * @param buffer - buffer to fill
    * @param setf - value to set all buffer entries to
    */
    public IntBuffer fillIntBuffer(IntBuffer buffer, int setf) {
        while(buffer.remaining() != 0){buffer.put(setf);}
        buffer.rewind();
        return buffer;
    }
    
    /**
    *       fillIntBuffer fills an int buffer with given value.
    * 
    * @param buffer - buffer to fill
    * @param setf - values to set all buffer entries to
    */
    public IntBuffer fillIntBuffer(IntBuffer buffer, int[] s) {
        int i=0; 
        while(buffer.remaining() != 0){buffer.put(s[i]);i++;}
        buffer.rewind();
        return buffer;
    }

    /**
    *       fillIntBuffer fills an int buffer with given array repeatedly using a fill based
    *   on a modulus of array size.
    * 
    * @param buffer - buffer to fill
    * @param setf - values to set all buffer entries to
    */
    public IntBuffer fillIntBufferMod(IntBuffer buffer, int[] s) {
        int i=0; 
        int size = s.length;
        while(buffer.remaining() != 0){buffer.put(s[(i%size)]);i++;}
        buffer.rewind();
        return buffer;
    }
    
    /**
    *       fillFloatBuffer fills a float buffer with given value.
    * 
    * @param buffer - buffer to fill
    * @param setf - values to set all buffer entries to
    */
    public FloatBuffer fillFloatBuffer(FloatBuffer buffer, float[] s) {
        int i=0; 
        while(buffer.remaining() != 0){buffer.put(s[i]);i++;}
        buffer.rewind();
        return buffer;
    }

    /**
    *       fillFloatBuffer fills a float buffer with given array repeatedly using a fill based
    *   on a modulus of array size.
    * 
    * @param buffer - buffer to fill
    * @param setf - values to set all buffer entries to
    */
    public FloatBuffer fillFloatBufferMod(FloatBuffer buffer, float[] s) {
        int i=0; 
        int size = s.length;
        while(buffer.remaining() != 0){
            buffer.put(s[(i%size)]);i++;}
        buffer.rewind();
        return buffer;
    }
    
    
    /**
    *       fillFloatBuffer fills a float buffer with given value.
    * 
    * @param buffer - buffer to fill
    * @param setf - value to set all buffer entries to
    */
    public FloatBuffer fillFloatBuffer(FloatBuffer buffer, float setf) {
        while(buffer.remaining() != 0){buffer.put(setf);}
        buffer.rewind();
        return buffer;
    }
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial value for the buffer entries 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float s0,int readwrite,boolean fill){
        // Create clbuffer object
        CLBuffer<FloatBuffer> buffer;
        if(readwrite ==2){
            buffer = context.createFloatBuffer(size, WRITE_ONLY);
        }else if(readwrite==1){
            buffer = context.createFloatBuffer(size, READ_ONLY);
        }else{
            buffer = context.createFloatBuffer(size, READ_WRITE);
        }    
        
        // fill the buffer if required
        if(fill){
            System.out.println("Filling Buffer");
        buffer.use(fillFloatBuffer(buffer.getBuffer(),s0));}
        
        
        // Initialize buffer type array if not already
        ArrayList<CLBuffer<FloatBuffer>> kbuff = this.flBuffers.get(kernelname);
        if(kbuff == null){ 
            System.err.println("Initializing float kernel buffers with value "+s0+" .");
            kbuff = new  ArrayList<CLBuffer<FloatBuffer>>();}
        
        // push buffer into buffers list       
        kbuff.add(argn, buffer);
        flBuffers.put(kernelname, kbuff);
    }    

    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float[] s0,int readwrite,boolean fill){
        createFloatBuffer(kernelname,argn,size,s0,readwrite,fill,false);
    }
    
    /**
    *       createFloatBuffer creates and fills an float buffer then adds it to 
    *   the list of float buffers for this kernel at given argument entry number. 
    * 
    * 
    * @param kernelname - kernel to add buffer for
    * @param argn - argument type entry number
    * @param size - size of buffer
    * @param s0 - initial values for the buffer 
    * @param readwrite - int (2,1,0) - 2 write only, 1 read only, 0 read and write
    * @param fill - true if filling buffer
    * @param modfill - fill using a modulus fill which fills based on mod of array size
    */    
    public void createFloatBuffer(String kernelname,int argn,int size, float[] s0,int readwrite,boolean fill, boolean modFill){
        // Create clbuffer object
        CLBuffer<FloatBuffer> buffer;
        if(readwrite ==2){
            buffer = context.createFloatBuffer(size, WRITE_ONLY);
        }else if(readwrite==1){
            buffer = context.createFloatBuffer(size, READ_ONLY);
        }else{
            buffer = context.createFloatBuffer(size, READ_WRITE);
        }    
        
        // fill the buffer if required
        if(fill){
        buffer.use(fillFloatBuffer(buffer.getBuffer(),s0));}else if(modFill){
          buffer.use(fillFloatBufferMod(buffer.getBuffer(),s0));
          
        }
        
        // Initialize buffer type array if not already
        ArrayList<CLBuffer<FloatBuffer>> kbuff = this.flBuffers.get(kernelname);
            if(kbuff == null){ 
            System.err.println("Initializing float kernel buffers.");
            kbuff = new  ArrayList<CLBuffer<FloatBuffer>>();}

        // push buffer into buffers list        
        kbuff.add(argn, buffer);
        flBuffers.put(kernelname, kbuff);
    }

    /**
    *       copyFLBufferAcrossKernel copies a buffer from source kernel float buffer list
    *   into destination kernel float buffer list.
    * 
    * @param skernel - source kernel name
    * @param sargn - source kernel float argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel float argument number
    */
    public void copyFlBufferAcrossKernel(String skernel,int sargn,String dkernel,int dargn){
        // get source buffer
        CLBuffer<FloatBuffer> inBuff = flBuffers.get(skernel).get(sargn);
        
        // get destination buffers
        ArrayList<CLBuffer<FloatBuffer>> kbuff = flBuffers.get(dkernel);

        // set destination buffers if not set 
        if(kbuff == null){ 
            System.err.println("Initializing float kernel buffers.");
            kbuff = new  ArrayList<CLBuffer<FloatBuffer>>();}
        
        // add source buffer to destination buffer set
        if(kbuff.size()< (dargn+1)){
        kbuff.add(dargn, inBuff);}else{
        kbuff.set(dargn, inBuff);
        }
        flBuffers.put(dkernel, kbuff);
    }
    
    /**
    *       copyIntBufferAcrossKernel copies a buffer from source kernel int buffer list
    *   into destination kernel int buffer list.
    * 
    * @param skernel - source kernel name
    * @param sargn - source kernel int argument number
    * @param dkernel - destination kernel name
    * @param dargn - destination kernel int argument number
    */
    public void copyIntBufferAcrossKernel(String skernel,int sargn,String dkernel,int dargn){
        // get source buffer
        CLBuffer<IntBuffer> inBuff = intBuffers.get(skernel).get(sargn);
        
        // get destination buffers
        ArrayList<CLBuffer<IntBuffer>> kbuff = intBuffers.get(dkernel);

        // set destination buffers if not set 
        if(kbuff == null){ 
            System.err.println("Initializing float kernel buffers.");
            kbuff = new  ArrayList<CLBuffer<IntBuffer>>();}
        
        // add source buffer to destination buffer set
        kbuff.add(dargn, inBuff);
        intBuffers.put(dkernel, kbuff);
    }
    
    /**
    *       createIntArg creates an int and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of int
    */
    public void createIntArg(String kernelname,int argn, int val){
        ArrayList<Integer> iargs = intArgs.get(kernelname);
        if(iargs==null){
            System.err.println("Initializing single int arg kernel.");
            iargs = new ArrayList<Integer>();
        }
        
        if(iargs.size()< (argn+1)){
        iargs.add(argn,val);}else{
            iargs.set(argn,val);
        }
        intArgs.put(kernelname,iargs);
    }

    /**
    *       createLongArg creates an Long and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of Long
    */
    public void createLongArg(String kernelname,int argn, long val){
        ArrayList<Long> iargs = longArgs.get(kernelname);
        if(iargs==null){
            System.err.println("Initializing single int arg kernel.");
            iargs = new ArrayList<Long>();
        }
        if(iargs.size()< (argn+1)){
        iargs.add(argn,val);}else{
            iargs.set(argn,val);
        }
        longArgs.put(kernelname,iargs);
    }

    
    /**
    *       getIntArg get an int from the kernel set.
    * 
    * @param kernelname - name of kernel to get int from
    * @param argn - argument type entry number
    */
    public int getIntArg(String kernelname,int argn){
        return intArgs.get(kernelname).get(argn);
    }
    
    /**
    *       getLongArg get a long from the kernel set.
    * 
    * @param kernelname - name of kernel to get long from
    * @param argn - argument type entry number
    */
    public long getLongArg(String kernelname,int argn){
        return longArgs.get(kernelname).get(argn);
    }
    
    /**
    *       getFloatArg get a float from the kernel set.
    * 
    * @param kernelname - name of kernel to get float from
    * @param argn - argument type entry number
    */
    public float getFloatArg(String kernelname,int argn){
        return floatArgs.get(kernelname).get(argn);
    }
    
    /**
    *       createFloatArg creates an float and adds it to the arguments for kernel.
    * 
    * @param kernelname - name of kernel to add argument to list
    * @param argn - argument type entry number
    * @param val - initial value of float
    */
    public void createFloatArg(String kernelname,int argn, float val){
        ArrayList<Float> fargs = floatArgs.get(kernelname);
        if(fargs==null){
            System.err.println("Initializing single float arg kernel.");
            fargs = new ArrayList<Float>();
        }
        
        fargs.add(argn,val);
        floatArgs.put(kernelname,fargs);
    }
        
    /**
    *       readFile converts a file into a string. Useful for the building of 
    *   OpenCL kernel.
    * 
    * @param filename - filename of file to convert.
    * @return String version of file
    */
    private String readFile(String filename) {
        File f = new File(filename);
        try {
            byte[] bytes = Files.readAllBytes(f.toPath());
            return new String(bytes,"UTF-8");
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        return "";
    }

    /**
    *       getKernelIOTypes reads the given kernels source and parses out a
    *   list of arguments to use for type checking.
    * 
    * @param kernelname - kernel to check arguments
    * @param source - source of kernel
    * @return array of arguments required
    */
    private ArrayList<String> getKernelIOTypes(String kernelname, String source){
        
        //parse io types
        String[] src=source.split("\\(");
        src = src[1].split("\\)");
        ArrayList<String> types = new ArrayList<String>();
        String[] typeSet = src[0].split(",");
        String proc,type="";
        boolean initFlBuff=false;
        boolean initIntBuff=false;
        boolean initFlArg=false;
        boolean initIntArg=false;
        
        // Parse kernel IO arguments into buffers that will be needed
        for(int i=0;i<typeSet.length;i++){
            proc = typeSet[i];
            type="";
            
            if(proc.contains("global")){
                type = "global "+type;
            }else if(proc.contains("local")){
                type = "local "+type;
            } 

            if(proc.contains("const")){
                type = "const "+type;
            }
            
            if(proc.contains("float")){
               type = type+"float";
            }else if(proc.contains("int")){
                type = type+"int";
           }else if(proc.contains("long")){
                type = type+"long";
            } 
            
            if(proc.contains("*")){
                type = type+" buffer";
            }
            
            //System.out.println("Adding Argument Type: "+type);
            types.add(i, type);

            // initialize device buffers and arguments in ideal way
            if(type.contains("buffer")&&type.contains("float")&& !initFlBuff){
                ArrayList<CLBuffer<FloatBuffer>> push = new  ArrayList<CLBuffer<FloatBuffer>>();
                flBuffers.put(kernelname, push);
            }else if(type.contains("buffer")&&type.contains("int")&& !initIntBuff){
                ArrayList<CLBuffer<IntBuffer>> push = new  ArrayList<CLBuffer<IntBuffer>>();
                intBuffers.put(kernelname, push);
            }else if(!type.contains("buffer")&&type.contains("int")&& !initIntArg){
                ArrayList<Integer> push = new  ArrayList<Integer>();
                intArgs.put(kernelname, push);
            }else if(!type.contains("buffer")&&type.contains("long")&& !initIntArg){
                ArrayList<Long> push = new  ArrayList<Long>();
                longArgs.put(kernelname, push);
            }else if(!type.contains("buffer")&&type.contains("float")&& !initFlBuff){
                ArrayList<Float> push = new  ArrayList<Float>();
                floatArgs.put(kernelname, push);
            }
        }
        return types;
    }
    
    /**
    *       assertKernelBuffersMade checks if all required buffers have been made.
    * 
    * @param kernelname - kernel to check buffers 
    * @return true if buffers are all present
    */
    private boolean assertKernelBuffersMade(String kernelname){
        boolean made = false;
        int total = argTypes.get(kernelname).size();
        int count = 0;
        
        // count the amount of arguments
        if(intBuffers.get(kernelname) != null){count += intBuffers.get(kernelname).size();}
        if(flBuffers.get(kernelname) != null){count += flBuffers.get(kernelname).size();}
        if(floatArgs.get(kernelname) != null){count += floatArgs.get(kernelname).size();}
        if(intArgs.get(kernelname) != null){count += intArgs.get(kernelname).size();}
        if(longArgs.get(kernelname) != null){count += longArgs.get(kernelname).size();}

        if(count == total){made=true;}else{
        System.err.println("ERROR: ALL KERNELS BUFFERS HAVE NOT BEEN INITIALIZED. KERNEL: "+kernelname
                +"----| kernel number of arguments: "+total+"    arguments initialized: "+count);}
        return made;
    }
    
    /**
    *           getIntBufferAsArray retrieves the device buffer and returns it
    *   into an array which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return integer array version of buffer
    */
    public int[] getIntBufferAsArray(String kernelname,int argn, int size,boolean print){
        CLBuffer<IntBuffer> buffer=intBuffers.get(kernelname).get(argn);
        //System.out.println("Buffer object is "+buffer);
        queue.putReadBuffer(buffer, true);
        
        int[] arr = new int[size];

        // push the buffer into array
        for(int i = 0; i < size; i++){
                arr[i]=buffer.getBuffer().get();
                if(print){System.out.println("i: "+i+"   val: "+arr[i]);}
        }

        // Need to rewind to start at same position after reads especially if partial
        buffer.getBuffer().rewind();
        
        return arr;
    }

    /**
    *           getFloatBufferAsArray retrieves the device buffer and returns it
    *   into an array which can be used in host.
    * 
    * @param kernelname - kernel to retrieve buffer from
    * @param argn - argument number 
    * @param size - size of buffer to retrieve 
    * @param print - true if printing out retrieved values
    * @return float array version of buffer
    */
    public float[] getFloatBufferAsArray(String kernelname,int argn, int size,boolean print){
        CLBuffer<FloatBuffer> buffer=flBuffers.get(kernelname).get(argn);
        queue.putReadBuffer(buffer, true);
        float[] arr = new float[size];
        
        // push the buffer into array
        for(int i = 0; i < size; i++){
                arr[i]=buffer.getBuffer().get();
                if(print){System.out.println("i: "+i+"   val: "+arr[i]);}
        }
        
        // Need to rewind to start at same position after reads especially if partial
        buffer.getBuffer().rewind();
        
        return arr;
    }
    
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    */
    public void setKernelArg(String kernelname){
        setKernelArg(kernelname,false);
    }
    
    /**
    *       setKernelArg sets the arguments of the given kernel by using 
    *   the arguments parsed from source.
    * 
    * @param kernelname - kernel to set arguments for
    * @param setPreviously - true if already set previously
    */
    public void setKernelArg(String kernelname, boolean setPrev){
        ArrayList<String> types = argTypes.get(kernelname);
        int intBuffInd = 0;
        int flBuffInd = 0;
        int intInd = 0;
        int flInd = 0;
        int longInd = 0;
        int Ind = 0;
        // Get kernel and all arguments
        CLKernel kernel = kernels.get(kernelname);
        ArrayList<CLBuffer<FloatBuffer>> floatBuff = flBuffers.get(kernelname);
        ArrayList<CLBuffer<IntBuffer>> intBuff = intBuffers.get(kernelname);
        ArrayList<Float> floatArg = floatArgs.get(kernelname);
        ArrayList<Integer> intArg = intArgs.get(kernelname);
        ArrayList<Long> longArg = longArgs.get(kernelname);
        
        // Set the arguments using the argument list
        for(int i=0;i<types.size();i++){
            String type = types.get(i);
            
            if(type.contains("int")&& type.contains("buffer") ){
                if(setPrev){
                    kernel.setArg(Ind,intBuff.get(intBuffInd));
                }else{
                kernel.putArg(intBuff.get(intBuffInd));}
                intBuffInd++;
            }else if(type.contains("float")&& type.contains("buffer")){
                if(setPrev){
                    kernel.setArg(Ind,floatBuff.get(flBuffInd));
                }else{
                kernel.putArg(floatBuff.get(flBuffInd));}                
                flBuffInd++;
            }else if(type.contains("float")&& !(type.contains("buffer"))){
                 if(setPrev){
                    kernel.setArg(Ind,floatArg.get(flInd));                    
                 }else{         
                     kernel.putArg(floatArg.get(flInd));
                 }
                                 
                flInd++;
            }else if(type.contains("int")&& !(type.contains("buffer"))){
                if(setPrev){
                      kernel.setArg(Ind,intArg.get(intInd));                    
                 }else{
                      kernel.putArg(intArg.get(intInd));
                  }                
                intInd++;    
            }else if(type.contains("long")&& !(type.contains("buffer"))){
                if(setPrev){
                    kernel.setArg(Ind,longArg.get(longInd));                    
                }else{
                    kernel.putArg(longArg.get(longInd));
                }                
                longInd++;    
            }else{System.err.println("ERROR: UNCATEGORIZED ARGUMENT");}
            Ind++;
        }
    }
    
    /**
    *       getDeviceUsedMB determines the amount of MB used by the kernel
    *   in the OpenCL device
    * 
    * 
    * @param kernelname - kernel name 
    * @return MB used by kernel
    */
    public int getDeviceUsedMB(String kernelname){
        float mbused=0;
        
        ArrayList<CLBuffer<FloatBuffer>> kbuff = flBuffers.get(kernelname);
        ArrayList<CLBuffer<IntBuffer>> ibuff = intBuffers.get(kernelname);

        // integer buffers b used
        for(int i=0;i<ibuff.size();i++){
            mbused = ibuff.get(i).getCLSize() +mbused;
        }

        // float buffers b used
        for(int i=0;i<kbuff.size();i++){
            mbused = kbuff.get(i).getCLSize() +mbused;
        }
        
        // make into mb
        mbused = mbused/1000000;
        
        return (int) mbused;
    }
    
    /**
    *       closeOpenCL releases the OpenCL context.
    */
    public void closeOpenCL(){
        context.release();
    }
    
    public static void main(String[] args) throws IOException {
        OpenCLHandler clhandle = new OpenCLHandler();

        int elementCount = 1000;                                  // Length of arrays to process
        int lsize = 1;  // Local work size dimensions
        int gsize = 1000;   // rounded up to the nearest multiple of the localWorkSize

        String kernel = "vector_add";

        clhandle.initializeOpenCL("GPU");

        clhandle.createKernel("", kernel);

        clhandle.createFloatBuffer(kernel, 0, 1000, 3.0f, 1, true);
        clhandle.createFloatBuffer(kernel, 1, 1000, 3.2f, 1, true);     
        clhandle.createFloatBuffer(kernel, 2, 1000, 2.1f, 2, false);
        clhandle.createIntArg(kernel, 0,1000);
        //clhandle.getFloatBufferAsArray(kernel, 2, 10, true);

        clhandle.setKernelArg(kernel);
        // The moving of the buffers into the device doesnt happen until the queue runs.
        
        clhandle.runKernel(kernel, gsize, lsize);

        clhandle.getFloatBufferAsArray(kernel, 2, 10, true);            

        clhandle.closeOpenCL();
     }
}
