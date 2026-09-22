// owl K1:nvrtc 无 libc 头路径,自带最小 stdint shim(仅本模块编译用;
// 类型宽度按 LP64 固定,与 driver ABI 一致)
#pragma once
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef short int16_t;
typedef unsigned short uint16_t;
typedef int int32_t;
typedef unsigned int uint32_t;
typedef long long int64_t;
typedef unsigned long long uint64_t;
