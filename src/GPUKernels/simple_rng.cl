__kernel void simple_rng(long seed1 , long seed2 , __global float* c)
{
  unsigned int m_z=521288629;
  unsigned int m_w=362436069;
  // get the index of the test we are performing
  int index = get_global_id(0);
  
  ulong oneCnt=0;
 
  // set the seed for the random generator
  m_z = m_z+(seed1+index);
  m_w = m_w+(seed2+index);
  unsigned int rnd;

  // Generate the random numbers and count the bits
    m_z = 36969 * (m_z & 65535) + (m_z >> 16);
    m_w = 18000 * (m_w & 65535) + (m_w >> 16);
    rnd = (m_z << 16) + (m_w & 65535);

  
  // 4294967295 is the largest unsigned int
  float outRan = ((float)rnd)/(4294967295);
  c[index] = outRan; 
}