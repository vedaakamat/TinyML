#include "imu.h"

/*---------------------------------------------------------------------------
 * External Variables
 *---------------------------------------------------------------------------*/

extern I2C_HandleTypeDef hi2c2;

/*---------------------------------------------------------------------------
 * Private Variables
 *---------------------------------------------------------------------------*/

static stmdev_ctx_t dev_ctx;

/*---------------------------------------------------------------------------
 * Platform Write Function
 *---------------------------------------------------------------------------*/

static int32_t Platform_Write(void *handle,
                              uint8_t reg,
                              uint8_t *bufp,
                              uint16_t len)
{
    if (HAL_I2C_Mem_Write(&hi2c2,
                          LSM6DSL_I2C_ADD_L,
                          reg,
                          I2C_MEMADD_SIZE_8BIT,
                          bufp,
                          len,
                          HAL_MAX_DELAY) != HAL_OK)
    {
        return -1;
    }

    return 0;
}

/*---------------------------------------------------------------------------
 * Platform Read Function
 *---------------------------------------------------------------------------*/

static int32_t Platform_Read(void *handle,
                             uint8_t reg,
                             uint8_t *bufp,
                             uint16_t len)
{
    if (HAL_I2C_Mem_Read(&hi2c2,
                         LSM6DSL_I2C_ADD_L,
                         reg,
                         I2C_MEMADD_SIZE_8BIT,
                         bufp,
                         len,
                         HAL_MAX_DELAY) != HAL_OK)
    {
        return -1;
    }

    return 0;
}

/*---------------------------------------------------------------------------
 * Platform Delay
 *---------------------------------------------------------------------------*/

static void Platform_Delay(uint32_t ms)
{
    HAL_Delay(ms);
}

/*---------------------------------------------------------------------------
 * IMU Initialization
 *---------------------------------------------------------------------------*/

int32_t IMU_Init(void)
{
    uint8_t id;

    dev_ctx.write_reg = Platform_Write;
    dev_ctx.read_reg  = Platform_Read;
    dev_ctx.mdelay    = Platform_Delay;
    dev_ctx.handle    = NULL;

    if (lsm6dsl_device_id_get(&dev_ctx, &id) != 0)
    {
        return -1;
    }

    if (id != LSM6DSL_ID)
    {
        return -1;
    }

    lsm6dsl_reset_set(&dev_ctx, PROPERTY_ENABLE);
    HAL_Delay(50);

    lsm6dsl_block_data_update_set(&dev_ctx, PROPERTY_ENABLE);

    lsm6dsl_auto_increment_set(&dev_ctx, PROPERTY_ENABLE);

    lsm6dsl_xl_full_scale_set(&dev_ctx, LSM6DSL_2g);

    lsm6dsl_xl_data_rate_set(&dev_ctx, LSM6DSL_XL_ODR_104Hz);

    lsm6dsl_gy_full_scale_set(&dev_ctx, LSM6DSL_250dps);

    lsm6dsl_gy_data_rate_set(&dev_ctx, LSM6DSL_GY_ODR_104Hz);

    return 0;
}

/*---------------------------------------------------------------------------
 * Read Accelerometer
 *---------------------------------------------------------------------------*/

int32_t IMU_Read_Accel(float *ax,
                       float *ay,
                       float *az)
{
    int16_t raw[3];

    if (lsm6dsl_acceleration_raw_get(&dev_ctx, raw) != 0)
    {
        return -1;
    }

    *ax = lsm6dsl_from_fs2g_to_mg(raw[0]) / 1000.0f;
    *ay = lsm6dsl_from_fs2g_to_mg(raw[1]) / 1000.0f;
    *az = lsm6dsl_from_fs2g_to_mg(raw[2]) / 1000.0f;

    return 0;
}

/*---------------------------------------------------------------------------
 * Read Gyroscope
 *---------------------------------------------------------------------------*/

int32_t IMU_Read_Gyro(float *gx,
                      float *gy,
                      float *gz)
{
    int16_t raw[3];

    if (lsm6dsl_angular_rate_raw_get(&dev_ctx, raw) != 0)
    {
        return -1;
    }

    *gx = lsm6dsl_from_fs250dps_to_mdps(raw[0]) / 1000.0f;
    *gy = lsm6dsl_from_fs250dps_to_mdps(raw[1]) / 1000.0f;
    *gz = lsm6dsl_from_fs250dps_to_mdps(raw[2]) / 1000.0f;

    return 0;
}
