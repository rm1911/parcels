#include <stdio.h>
#include <math.h>

typedef struct
{
  int xdim, ydim, tdim, tidx;
  float *lon, *lat;
  double *time;
  float ***data;
} CField;


/* Local linear search to update grid index */
static inline int search_linear_float(float x, int i, int size, float *xvals)
{
    while (i < size-1 && x > xvals[i+1]) ++i;
    while (i > 0 && x < xvals[i]) --i;
    return i;
}

/* Local linear search to update time index */
static inline int search_linear_double(double t, int i, int size, double *tvals)
{
    while (i < size-1 && t > tvals[i+1]) ++i;
    while (i > 0 && t < tvals[i]) --i;
    return i;
}

/* Bilinear interpolation routine for 2D grid */
static inline float spatial_interpolation_bilinear(float x, float y, int i, int j, int ydim,
                                                   float *lon, float *lat, float **f_data, char name)
{
  /* Cast data array into data[lat][lon] as per NEMO convention */
  float (*data)[ydim] = (float (*)[ydim]) f_data;

  float sw = data[i][j];
  float nw = data[i+1][j];
  float se = data[i][j+1];
  float ne = data[i+1][j+1];
  float swap;

  if (name == 'V') {
    float swap = nw;
    nw = se;
    se = swap;
  }

  float wsw;
  float wnw;
  float ese;
  float ene;

  float repel = 1.01;
  float big_repel = 1.5;

  if (!(isnan(sw) || isnan(nw) || isnan(se) || isnan(ne)))
    goto end;			//RETURN

  if (name == 'U' || name == 'V') {
    if (isnan(sw) && isnan(nw) && isnan(se) && isnan(ne)) {
      if (name == 'U') {
        wsw = data[i][j-1];
        wnw = data[i+1][j-1];
        ese = data[i][j+2];
        ene = data[i+1][j+2];
      }
      if (name == 'V') {
        wsw = data[i-1][j];
        wnw = data[i-1][j+1];
        ese = data[i+2][j];
        ene = data[i+2][j+1];
      }

      if ((y - lat[j] < lat[j+1] - y && name == 'U') || (x - lon[i] < lon[i+1] - x && name == 'V')) {
        if (!isnan(wsw))
          sw = se = -fabs(wsw) * big_repel;
        else
          sw = se = -fabs(wnw) * big_repel;
        if (!isnan(wnw))
          nw = ne = -fabs(wnw) * big_repel;
        else
          nw = ne = -fabs(wsw) * big_repel;
      } else {
        if (!isnan(ese))
          sw = se = fabs(ese) * big_repel;
        else
          sw = se = fabs(ene) * big_repel;
        if (!isnan(ene))
          nw = ne = fabs(ene) * big_repel;
        else
          nw = ne = fabs(ese) * big_repel;
      }
      goto end;			//RETURN
    }

    if (isnan(sw)) {
      if (isnan(nw)) {
        if (isnan(se)) {
          sw = nw = fabs(ne) * repel;
          se = ne;
        } else if (isnan(ne)) {
          sw = nw = fabs(se) * repel;
          ne = se;
        } else {
          sw = fabs(se) * repel;
          nw = fabs(ne) * repel;
        }
        goto end;		//RETURN
      } else
        sw = nw;
    } else if (isnan(nw))
      nw = sw;
    if (isnan(se)) {
      if (isnan(ne)) {
        se = -fabs(sw) * repel;
        ne = -fabs(nw) * repel;
      } else
        se = ne;
    } else if (isnan(ne))
      ne = se;

  } else {
    if (isnan(sw))
      sw = 0;
    if (isnan(nw))
      nw = 0;
    if (isnan(se))
      se = 0;
    if (isnan(ne))
      ne = 0;
  }

  end:
  if (name == 'V') {
    swap = nw;
    nw = se;
    se = swap;
  }

  return (sw * (lat[j+1] - y) * (lon[i+1] - x)
        + se * (y - lat[j]) * (lon[i+1] - x)
        + nw * (lat[j+1] - y) * (x - lon[i])
        + ne * (y - lat[j]) * (x - lon[i]))
        / ((lat[j+1] - lat[j]) * (lon[i+1] - lon[i]));
}

/* Linear interpolation along the time axis */
static inline float temporal_interpolation_linear(float x, float y, int xi, int yi,
                                                  double time, CField *f, char name)
{
  /* Cast data array intp data[time][lat][lon] as per NEMO convention */
  float (*data)[f->xdim][f->ydim] = (float (*)[f->xdim][f->ydim]) f->data;
  float f0, f1;
  double t0, t1;
  int i = xi, j = yi;
  /* Identify grid cell to sample through local linear search */
  i = search_linear_float(x, i, f->xdim, f->lon);
  j = search_linear_float(y, j, f->ydim, f->lat);
  /* Find time index for temporal interpolation */
  f->tidx = search_linear_double(time, f->tidx, f->tdim, f->time);
  if (f->tidx < f->tdim-1 && time > f->time[f->tidx]) {
    t0 = f->time[f->tidx]; t1 = f->time[f->tidx+1];
    f0 = spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[f->tidx]), name);
    f1 = spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[f->tidx+1]), name);
    return f0 + (f1 - f0) * (float)((time - t0) / (t1 - t0));
  } else {
    return spatial_interpolation_bilinear(x, y, i, j, f->ydim, f->lon, f->lat, (float**)(data[f->tidx]), name);
  }
}
