#include "ggml-cpu/vec.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu.h"
#include "ggml-quants.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <vector>
#include <string>

static std::string output_dir;
static const char * rope_table_path;

static void write_file(const char * name, const void * data, size_t bytes) {
    std::string path=output_dir+"/"+name;
    FILE * f = fopen(path.c_str(), "wb");
    if (!f || fwrite(data, bytes, 1, f) != 1 || fclose(f)) std::abort();
}

static size_t kv_index(int kind, int position, int head, int component) {
    return ((((kind*512 + position)*2 + head)*64) + component);
}

static void attention(const float * query, const float * kv, float * output, float * scores,
                      int current, int span) {
    const int positions = current + 1;
    for (int qh = 0; qh < 14; ++qh) {
        const int kh = qh/7;
        float maximum = -INFINITY;
        for (int p = 0; p < positions; ++p) {
            float dot = 0;
            ggml_vec_dot_f32(64, &dot, 0, query+qh*64, 0, kv+kv_index(0,p,kh,0), 0, 1);
            scores[p] = dot*0.125f;
            maximum = std::max(maximum, scores[p]);
        }
        for (int p = positions; p < span; ++p) scores[p] = -INFINITY;
        double sum = ggml_vec_soft_max_f32(span, scores, scores, maximum);
        const float inv = (float)(1.0/sum);
        for (int p = 0; p < positions; ++p) scores[p] *= inv;
        for (int c = 0; c < 64; ++c) {
            std::vector<float> values(positions);
            for (int p = 0; p < positions; ++p) values[p]=kv[kv_index(1,p,kh,c)];
            float total=0;
            ggml_vec_dot_f32(positions, &total, 0, scores, 0, values.data(), 0, 1);
            output[qh*64+c] = total;
        }
    }
}

int main(int argc, char ** argv) {
    if(argc!=3)return 64;
    output_dir=argv[1];
    rope_table_path=argv[2];
    ggml_cpu_init();
    _mm_setcsr(0x1f80);
    static_assert(sizeof(block_q4_0)==18 && sizeof(block_q8_0)==34);
    std::vector<float> qin(32*6, 0);
    qin[32]=1; qin[33]=-1; qin[34]=1; qin[35]=-1;
    qin[64]=127; qin[65]=.5f; qin[66]=-.5f; qin[67]=1.5f; qin[68]=-1.5f;
    qin[96]=0x1p-23f; qin[97]=0x1p-24f; qin[98]=0x1p-25f;
    qin[128]=126.5f; qin[129]=-126.5f; qin[130]=63.25f;
    qin[160]=-0.0f; qin[161]=0.0f;
    block_q8_0 qblocks[6]; memset(qblocks, 0xa5, sizeof(qblocks));
    quantize_row_q8_0_ref(qin.data(), qblocks, qin.size());
    write_file("q8.bin", qblocks, sizeof(qblocks));

    float minput[64];
    for (int i=0;i<64;++i) minput[i]=(float)((i%31)-15);
    minput[31]=127; minput[63]=-127;
    block_q8_0 act[2]; quantize_row_q8_0_ref(minput, act, 64);
    block_q4_0 q4[2]{}; block_q8_0 q8[2]{};
    for (int b=0;b<2;++b) {
        q4[b].d=0x3c00;
        for(int i=0;i<16;++i) q4[b].qs[i]=(uint8_t)(((15-i)<<4)|i);
        q8[b].d=0x3c00;
        for(int i=0;i<32;++i) q8[b].qs[i]=(int8_t)((i%17)-8);
    }
    float q4total=0, q8total=0;
    ggml_vec_dot_q4_0_q8_0_generic(64,&q4total,0,q4,0,act,0,1);
    ggml_vec_dot_q8_0_q8_0_generic(64,&q8total,0,q8,0,act,0,1);
    float dots[2]={q4total,q8total}; write_file("dots.bin",dots,sizeof(dots));

    std::vector<float> rms(896), weight(896), rout(896);
    for(int i=0;i<896;++i){rms[i]=(float)((i%17)-8)/8;weight[i]=(float)(i%7+1)/4;}
    double sum=0;for(float x:rms)sum+=(double)(x*x);
    float scale=1.0f/sqrtf((float)(sum/896)+1.0e-6f);
    for(int i=0;i<896;++i)rout[i]=rms[i]*scale*weight[i];
    write_file("rms.bin",rout.data(),rout.size()*4);

    std::vector<float> gate(4864),up(4864),product(4864);
    for(int i=0;i<4864;++i){gate[i]=(float)((i%33)-16)/16;up[i]=(float)((i%19)-9)/8;}
    ggml_vec_swiglu_f32(4864,product.data(),gate.data(),up.data());
    write_file("swiglu.bin",product.data(),product.size()*4);

    std::vector<float> query(896),kv(12582912/4,0),out(896),scores(512);
    for(int i=0;i<896;++i)query[i]=(float)((i%23)-11)/16;
    for(int p=0;p<=256;++p)for(int h=0;h<2;++h)for(int c=0;c<64;++c){
        kv[kv_index(0,p,h,c)]=(float)(((p+c+h)%29)-14)/32;
        kv[kv_index(1,p,h,c)]=(float)(((p*3+c+h)%31)-15)/32;
    }
    attention(query.data(),kv.data(),out.data(),scores.data(),2,256);
    write_file("attention-256.bin",out.data(),out.size()*4);
    write_file("scores-256.bin",scores.data(),256*4);
    std::rotate(query.begin(),query.begin()+7,query.end());
    attention(query.data(),kv.data(),out.data(),scores.data(),256,512);
    write_file("attention-512.bin",out.data(),out.size()*4);
    write_file("scores-512.bin",scores.data(),512*4);

    std::vector<float> table(512*32*2),rope(64);
    FILE * rf=fopen(rope_table_path,"rb");
    if(!rf || fread(table.data(),table.size()*4,1,rf)!=1 || fclose(rf))std::abort();
    for(int i=0;i<64;++i)rope[i]=(float)(i-32)/16;
    for(int pair=0;pair<32;++pair){
        float left=rope[pair],right=rope[pair+32];
        float cosine=table[(511*32+pair)*2],sine=table[(511*32+pair)*2+1];
        rope[pair]=left*cosine-right*sine;
        rope[pair+32]=left*sine+right*cosine;
    }
    write_file("rope.bin",rope.data(),rope.size()*4);
}
