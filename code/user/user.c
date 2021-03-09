
/*
 * This file is part of the micropython-ulab project,
 *
 * https://github.com/v923z/micropython-ulab
 *
 * The MIT License (MIT)
 *
 * Copyright (c) 2020 Zoltán Vörös
*/

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "py/obj.h"
#include "py/runtime.h"
#include "py/misc.h"
#include "user.h"

#if ULAB_USER_MODULE

//| """This module should hold arbitrary user-defined functions."""
//|

static mp_obj_t user_square(mp_obj_t arg) {
    // the function takes a single dense ndarray, and calculates the
    // element-wise square of its entries

    // raise a TypeError exception, if the input is not an ndarray
    if(!MP_OBJ_IS_TYPE(arg, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("input must be an ndarray"));
    }
    ndarray_obj_t *ndarray = MP_OBJ_TO_PTR(arg);

    // make sure that the input is a dense array
    if(!ndarray_is_dense(ndarray)) {
        mp_raise_TypeError(translate("input must be a dense ndarray"));
    }

    // if the input is a dense array, create `results` with the same number of
    // dimensions, shape, and dtype
    ndarray_obj_t *results = ndarray_new_dense_ndarray(ndarray->ndim, ndarray->shape, ndarray->dtype);

    // since in a dense array the iteration over the elements is trivial, we
    // can cast the data arrays ndarray->array and results->array to the actual type
    if(ndarray->dtype == NDARRAY_UINT8) {
        uint8_t *array = (uint8_t *)ndarray->array;
        uint8_t *rarray = (uint8_t *)results->array;
        for(size_t i=0; i < ndarray->len; i++, array++) {
            *rarray++ = (*array) * (*array);
        }
    } else if(ndarray->dtype == NDARRAY_INT8) {
        int8_t *array = (int8_t *)ndarray->array;
        int8_t *rarray = (int8_t *)results->array;
        for(size_t i=0; i < ndarray->len; i++, array++) {
            *rarray++ = (*array) * (*array);
        }
    } else if(ndarray->dtype == NDARRAY_UINT16) {
        uint16_t *array = (uint16_t *)ndarray->array;
        uint16_t *rarray = (uint16_t *)results->array;
        for(size_t i=0; i < ndarray->len; i++, array++) {
            *rarray++ = (*array) * (*array);
        }
    } else if(ndarray->dtype == NDARRAY_INT16) {
        int16_t *array = (int16_t *)ndarray->array;
        int16_t *rarray = (int16_t *)results->array;
        for(size_t i=0; i < ndarray->len; i++, array++) {
            *rarray++ = (*array) * (*array);
        }
    } else { // if we end up here, the dtype is NDARRAY_FLOAT
        mp_float_t *array = (mp_float_t *)ndarray->array;
        mp_float_t *rarray = (mp_float_t *)results->array;
        for(size_t i=0; i < ndarray->len; i++, array++) {
            *rarray++ = (*array) * (*array);
        }
    }
    // at the end, return a micrppython object
    return MP_OBJ_FROM_PTR(results);
}

MP_DEFINE_CONST_FUN_OBJ_1(user_square_obj, user_square);

static mp_obj_t back_sub(mp_obj_t _U, mp_obj_t _b) {

    if(!MP_OBJ_IS_TYPE(_U, &ulab_ndarray_type) || !MP_OBJ_IS_TYPE(_b, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("arguments must be ndarrays"));
    }

    ndarray_obj_t *U = MP_OBJ_TO_PTR(_U);
    ndarray_obj_t *b = MP_OBJ_TO_PTR(_b);
    
    if(!ndarray_is_dense(U) || !ndarray_is_dense(b)) {
        mp_raise_TypeError(translate("input must be a dense ndarray"));
    }

    ndarray_obj_t *x = ndarray_new_dense_ndarray(b->ndim, b->shape, b->dtype);
    mp_float_t *x_arr = (mp_float_t *)x->array;
    
    mp_float_t *U_arr = (mp_float_t *)U->array;
    mp_float_t *b_arr = (mp_float_t *)b->array;

    size_t U_rows = U->shape[ULAB_MAX_DIMS - 2];
    size_t U_cols = U->shape[ULAB_MAX_DIMS - 1];

    for (size_t i = U_rows - 1; i < U_rows; i--) {
        mp_float_t sum = 0.0;
        for (size_t j = i + 1; j < U_cols; j++) {
            sum += (*(U_arr + ((i * U_cols) + j)))
                        * (*(x_arr + j)); 
        }
        *(x_arr + i) = ((*(b_arr + i)) - sum) / (*(U_arr + ((i * U_cols) + i)));
    }

    return MP_OBJ_FROM_PTR(x);
}

MP_DEFINE_CONST_FUN_OBJ_2(user_back_sub_obj, back_sub);

static mp_obj_t forw_sub(mp_obj_t _L, mp_obj_t _b) {

    if(!MP_OBJ_IS_TYPE(_L, &ulab_ndarray_type) || !MP_OBJ_IS_TYPE(_b, &ulab_ndarray_type)) {
        mp_raise_TypeError(translate("arguments must be ndarrays"));
    }

    ndarray_obj_t *L = MP_OBJ_TO_PTR(_L);
    ndarray_obj_t *b = MP_OBJ_TO_PTR(_b);
    
    if(!ndarray_is_dense(L) || !ndarray_is_dense(b)) {
        mp_raise_TypeError(translate("input must be a dense ndarray"));
    }

    ndarray_obj_t *x = ndarray_new_dense_ndarray(b->ndim, b->shape, b->dtype);
    mp_float_t *x_arr = (mp_float_t *)x->array;
    
    mp_float_t *L_arr = (mp_float_t *)L->array;
    mp_float_t *b_arr = (mp_float_t *)b->array;

    size_t L_rows = L->shape[ULAB_MAX_DIMS - 2];
    size_t L_cols = L->shape[ULAB_MAX_DIMS - 1];

    for (size_t i = 0; i < L_rows; i++) {
        mp_float_t sum = 0.0;
        for (size_t j = 0; j < i; j++) {
            sum += (*(L_arr + ((i * L_cols) + j)))
                        * (*(x_arr + j)); 
        }
        *(x_arr + i) = ((*(b_arr + i)) - sum) / (*(L_arr + ((i * L_cols) + i)));
    }

    return MP_OBJ_FROM_PTR(x);
}

MP_DEFINE_CONST_FUN_OBJ_2(user_forw_sub_obj, forw_sub);

STATIC const mp_rom_map_elem_t ulab_user_globals_table[] = {
    { MP_OBJ_NEW_QSTR(MP_QSTR___name__), MP_OBJ_NEW_QSTR(MP_QSTR_user) },
    { MP_OBJ_NEW_QSTR(MP_QSTR_square), (mp_obj_t)&user_square_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_back_sub), (mp_obj_t)&user_back_sub_obj },
    { MP_OBJ_NEW_QSTR(MP_QSTR_forw_sub), (mp_obj_t)&user_forw_sub_obj },
};

STATIC MP_DEFINE_CONST_DICT(mp_module_ulab_user_globals, ulab_user_globals_table);

mp_obj_module_t ulab_user_module = {
    .base = { &mp_type_module },
    .globals = (mp_obj_dict_t*)&mp_module_ulab_user_globals,
};

#endif
