#ifndef __IMU_H
#define __IMU_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"
#include "lsm6dsl_reg.h"

int32_t IMU_Init(void);

int32_t IMU_Read_Accel(float *ax, float *ay, float *az);

int32_t IMU_Read_Gyro(float *gx, float *gy, float *gz);

#ifdef __cplusplus
}
#endif

#endif
