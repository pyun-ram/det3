/*
Note: This is from https://github.com/prclibo/kitti_eval/blob/master/evaluate_object_3d_offline.cpp
apt-get install libboost-dev
g++ -o tools/evaluate_object_3d_offline_carla tools/evaluate_object_3d_offline_carla.cpp
./tools/evaluate_object_3d_offline_carla groundtruth_dir result_dir
 */

#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>

#include <dirent.h>

#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/polygon.hpp>
#include <boost/geometry/geometries/adapted/c_array.hpp>

#include "mail.h"

BOOST_GEOMETRY_REGISTER_C_ARRAY_CS(cs::cartesian)

typedef boost::geometry::model::polygon<boost::geometry::model::d2::point_xy<double> > Polygon;


using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// holds the number of test images on the server
const int32_t N_TESTIMAGES = 7518;

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2};

// evaluation metrics: image, ground or 3D
enum METRIC{IMAGE=0, GROUND=1, BOX3D=2};

// evaluation parameter
const int32_t MIN_HEIGHT[3]     = {40, 25, 25};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3]  = {0, 1, 2};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[3] = {0.15, 0.3, 0.5}; // maximum truncation level of the groundtruth used for evaluation

// evaluated object classes
enum CLASSES{CAR=0, PEDESTRIAN=1, CYCLIST=2};
const int NUM_CLASS = 3;

// parameters varying per class
vector<string> CLASS_NAMES;
// the minimum overlap required for 2D evaluation on the image/ground plane and 3D evaluation
// const double MIN_OVERLAP[3][3] = {{0.7, 0.5, 0.5}, {0.5, 0.25, 0.25}, {0.5, 0.25, 0.25}};
const double MIN_OVERLAP[3][3] = {{0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}, {0.7, 0.5, 0.5}};

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;


// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("pedestrian");
  CLASS_NAMES.push_back("cyclist");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  double ry;
  double  t1, t2, t3;
  double h, w, l;
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  double  ry;
  double  t1, t2, t3;
  double  h, w, l;
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};


/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/
vector<int32_t> indices;


void apply_R(double M[][3], double R[][3], int32_t num_pts)
{
  for(int32_t i=0; i < num_pts; i++)
  {
    double x, y, z;
    x = M[i][0];
    y = M[i][1];
    z = M[i][2];
    M[i][0] = x * R[0][0] + y * R[0][1] + z * R[0][2];
    M[i][1] = x * R[1][0] + y * R[1][1] + z * R[1][2];
    M[i][2] = x * R[2][0] + y * R[2][1] + z * R[2][2];
  }
}

void apply_Tr(double M[][3], double *Tr, int32_t num_pts)
{
  for(int32_t i=0; i < num_pts; i++)
  {
    M[i][0] = M[i][0] + Tr[0];
    M[i][1] = M[i][1] + Tr[1];
    M[i][2] = M[i][2] + Tr[2];
  }
}

vector< vector<double> > get_bbox3dcorners(float* box3d)
{
  float h, w, l, x, y, z, ry;
  h = box3d[0];  w = box3d[1];  l = box3d[2];
  x = box3d[3];  y = box3d[4];  z = box3d[5];
  ry = box3d[6];

  double corners[8][3];
  double R[3][3];
  double Tr[3];
  for(int32_t i = 0; i < 8; i++)
    for(int32_t j = 0; j < 3; j++)
      corners[i][j] = 0;
  for(int32_t i = 0; i < 4; i++)
  {
    corners[i][0] = l/2.0;
    corners[i][1] = w/2.0;
  }
  corners[0][0] *=-1;
  corners[2][1] *=-1;
  corners[3][0] *=-1;
  corners[3][1] *=-1;

  for(int32_t i=0; i < 3; i++)
    for(int32_t j = 0; j < 3; j++)
      R[i][j] = 0;
  R[0][0] = cos(ry);  R[0][1] = -sin(ry);
  R[1][0] = sin(ry);  R[1][1] = cos(ry);
  R[2][2] = 1;
  Tr[0] = x;  Tr[1] = y;  Tr[2] = z;
  apply_R(corners, R, 4);
  apply_Tr(corners, Tr, 4);
  for(int32_t i = 0; i < 4; i++)
  {
    corners[i+4][0] = corners[i][0];
    corners[i+4][1] = corners[i][1];
    corners[i+4][2] = corners[i][2] + h;
  }

  vector<vector<double> > v_corners;
  for(int32_t i = 0; i < 8; i++)
  {
    vector<double> corner;
    corner.push_back(corners[i][0]);
    corner.push_back(corners[i][1]);
    corner.push_back(corners[i][2]);
    v_corners.push_back(corner);
  }

  return v_corners;
}

vector< vector<double> > imu2cam(vector< vector<double> > corners_imu)
{
  vector< vector<double> > corners_cam;
  for(int32_t i=0; i<corners_imu.size(); i++)
  {
    vector<double> corner_cam;
    double x,y,z;
    x = corners_imu[i][0];
    y = corners_imu[i][1];
    z = corners_imu[i][2];
    corner_cam.push_back(-y);
    corner_cam.push_back(-z);
    corner_cam.push_back(x);
    corners_cam.push_back(corner_cam);
  }
  return corners_cam;
}

vector< vector<double> > cam2imgplane(vector< vector<double> >corners)
{
  vector< vector<double> > corners_2d;
  double P[3][4] = {450, 0., 600, 0., 0., 450, 180, 0., 0., 0., 1, 0.};
  for(int32_t i=0; i < corners.size(); i++)
  {
    double x,y,z;
    vector<double> corner_2d;
    x = corners[i][0];
    y = corners[i][1];
    z = corners[i][2];
    corners[i][0] = P[0][0] * x + P[0][1] * y + P[0][2] * z + P[0][3];
    corners[i][1] = P[1][0] * x + P[1][1] * y + P[1][2] * z + P[1][3];
    corners[i][2] = P[2][0] * x + P[2][1] * y + P[2][2] * z + P[2][3] + 1e-6;
    corner_2d.push_back(corners[i][0] / corners[i][2]);
    corner_2d.push_back(corners[i][1] / corners[i][2]);
    corners_2d.push_back(corner_2d);
  }
  return corners_2d;
}

vector<double> find_min_max(double* v, const unsigned int n)
{
  double res_min=1e8;
  double res_max = -1e8;
  vector<double> res;
  for(int32_t i = 0; i < n; i++)
    if (v[i] < res_min)
      res_min=v[i];

  for(int32_t i = 0; i < n; i++)
    if (v[i] > res_max)
      res_max=v[i];

  res.push_back(res_min);
  res.push_back(res_max);
  return res;
}


vector<tDetection> loadDetections(string file_name, bool &compute_aos,
        vector<bool> &eval_image, vector<bool> &eval_ground,
        vector<bool> &eval_3d, bool &success) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return detections;
  }
  while (!feof(fp)) {
    tDetection d;
    double trash;
    char str[255];
    double truncation, occ, alpha, x1, y1, x2, y2, h, w, l, t1, t2, t3, ry, score;
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &truncation, &occ, &alpha,
                   &x1,   &y1,     &x2,    &y2,
                   &h,      &w,        &l,       &t1,
                   &t2,      &t3,        &ry , &score)==16) {

        // d.thresh = 1;
      d.box.type = str;
      d.box.alpha = alpha;
      d.w = l;
      d.l = h;
      d.h = w;
      d.t1 = -t2;
      d.t2 = -t3;
      d.t3 = t1;
      d.ry = -ry;
      d.thresh = score;
      if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0)
      {
        float box3d[7] = {h, w, l, t1, t2 ,t3, ry};
        vector< vector<double> > corners_imu = get_bbox3dcorners(box3d);
        vector< vector<double> > corners_cam = imu2cam(corners_imu);
        vector< vector<double> > corners_2d = cam2imgplane(corners_cam);

        double xs_2d[8], ys_2d[8];
        for(int32_t i = 0; i < 8; i++)
        {
          xs_2d[i] = corners_2d[i][0];
          ys_2d[i] = corners_2d[i][1];
        }
        vector<double> min_max_x, min_max_y;
        min_max_x = find_min_max(xs_2d, 8);
        min_max_y = find_min_max(ys_2d, 8);
        d.box.x1 = min_max_x[0];
        d.box.x2 = min_max_x[1];
        d.box.y1 = min_max_y[0];
        d.box.y2 = min_max_y[1];
      }
      else
      {
        d.box.x1 = x1;
        d.box.x2 = x2;
        d.box.y1 = y1;
        d.box.y2 = y2;
      }
      // cout<<d.box.type<<" "<< 0<<" "<< 0<<" "<< d.box.alpha<<" "
      //       <<d.box.x1<<" "<<d.box.y1<<" "<<d.box.x2<<" "<<d.box.y2<<" "
      //       <<d.h<<" "<<d.w<<" "<<d.l<<" "<<d.t1<<" "<<" "<<d.t2<<" "<<d.t3<<" "<<d.ry <<" "<<d.thresh<<endl;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha == -10)
        compute_aos = false;

      // a class is only evaluated if it is detected at least once
      for (int c = 0; c < NUM_CLASS; c++) {
        if (!strcasecmp(d.box.type.c_str(), CLASS_NAMES[c].c_str())) {
          if (!eval_image[c] && d.box.x1 >= 0)
            eval_image[c] = true;
          if (!eval_ground[c] && d.t1 != -1000)
            eval_ground[c] = true;
          if (!eval_3d[c] && d.t2 != -1000)
            eval_3d[c] = true;
          break;
        }
      }
    }
  }
  fclose(fp);
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    char str[255];
    double truncation, occ, alpha, x1, y1, x2, y2, h, w, l, t1, t2, t3, ry;
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &truncation, &occ, &alpha,
                   &x1,   &y1,     &x2,    &y2,
                   &h,      &w,        &l,       &t1,
                   &t2,      &t3,        &ry )==15) {
      g.box.type = str;
      g.truncation = truncation;
      g.occlusion = (int32_t)occ;
      g.box.alpha = alpha;
      g.w = l;
      g.l = h;
      g.h = w;
      g.t1 = -t2;
      g.t2 = -t3;
      g.t3 = t1;
      g.ry = -ry;
      if (x1 == 0 && x2 == 0 && y1 == 0 && y2 == 0)
      {
        float box3d[7] = {h, w, l, t1, t2 ,t3, ry};
        vector< vector<double> > corners_imu = get_bbox3dcorners(box3d);
        vector< vector<double> > corners_cam = imu2cam(corners_imu);
        vector< vector<double> > corners_2d = cam2imgplane(corners_cam);

        double xs_2d[8], ys_2d[8];
        for(int32_t i = 0; i < 8; i++)
        {
          xs_2d[i] = corners_2d[i][0];
          ys_2d[i] = corners_2d[i][1];
        }
        vector<double> min_max_x, min_max_y;
        min_max_x = find_min_max(xs_2d, 8);
        min_max_y = find_min_max(ys_2d, 8);
        g.box.x1 = min_max_x[0];
        g.box.x2 = min_max_x[1];
        g.box.y1 = min_max_y[0];
        g.box.y2 = min_max_y[1];
      }
      else
      {
        g.box.x1 = x1;
        g.box.x2 = x2;
        g.box.y1 = y1;
        g.box.y2 = y2;
      }
      // cout<<g.box.type<<" "<< g.truncation<<" "<< g.occlusion<<" "<< g.box.alpha<<" "
      //       <<g.box.x1<<" "<<g.box.y1<<" "<<g.box.x2<<" "<<g.box.y2<<" "
      //       <<g.h<<" "<<g.w<<" "<<g.l<<" "<<g.t1<<" "<<" "<<g.t2<<" "<<g.t3<<" "<<g.ry <<endl;
      groundtruth.push_back(g);
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double imageBoxOverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

inline double imageBoxOverlap(tDetection a, tGroundtruth b, int32_t criterion=-1){
  return imageBoxOverlap(a.box, b.box, criterion);
}

// compute polygon of an oriented bounding box
template <typename T>
Polygon toPolygon(const T& g) {
    using namespace boost::numeric::ublas;
    using namespace boost::geometry;
    matrix<double> mref(2, 2);
    mref(0, 0) = cos(g.ry); mref(0, 1) = sin(g.ry);
    mref(1, 0) = -sin(g.ry); mref(1, 1) = cos(g.ry);

    static int count = 0;
    matrix<double> corners(2, 4);
    double data[] = {g.l / 2, g.l / 2, -g.l / 2, -g.l / 2,
                     g.w / 2, -g.w / 2, -g.w / 2, g.w / 2};
    std::copy(data, data + 8, corners.data().begin());
    matrix<double> gc = prod(mref, corners);
    for (int i = 0; i < 4; ++i) {
        gc(0, i) += g.t1;
        gc(1, i) += g.t3;
    }

    double points[][2] = {{gc(0, 0), gc(1, 0)},{gc(0, 1), gc(1, 1)},{gc(0, 2), gc(1, 2)},{gc(0, 3), gc(1, 3)},{gc(0, 0), gc(1, 0)}};
    Polygon poly;
    append(poly, points);
    return poly;
}

// measure overlap between bird's eye view bounding boxes, parametrized by (ry, l, w, tx, tz)
inline double groundBoxOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double inter_area = in.empty() ? 0 : area(in.front());
    double union_area = area(un.front());
    double o;
    if(criterion==-1)     // union
        o = inter_area / union_area;
    else if(criterion==0) // bbox_a
        o = inter_area / area(dp);
    else if(criterion==1) // bbox_b
        o = inter_area / area(gp);

    return o;
}

// measure overlap between 3D bounding boxes, parametrized by (ry, h, w, l, tx, ty, tz)
inline double box3DOverlap(tDetection d, tGroundtruth g, int32_t criterion = -1) {
    using namespace boost::geometry;
    Polygon gp = toPolygon(g);
    Polygon dp = toPolygon(d);

    std::vector<Polygon> in, un;
    intersection(gp, dp, in);
    union_(gp, dp, un);

    double ymax = min(d.t2, g.t2);
    double ymin = max(d.t2 - d.h, g.t2 - g.h);

    double inter_area = in.empty() ? 0 : area(in.front());
    double inter_vol = inter_area * max(0.0, ymax - ymin);

    double det_vol = d.h * d.l * d.w;
    double gt_vol = g.h * g.l * g.w;

    double o;
    if(criterion==-1)     // union
        o = inter_vol / (det_vol + gt_vol - inter_vol);
    else if(criterion==0) // bbox_a
        o = inter_vol / det_vol;
    else if(criterion==1) // bbox_b
        o = inter_vol / gt_vol;

    return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty){

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Pedestrian") && !strcasecmp("Person_sitting", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Car") && !strcasecmp("Van", gt[i].box.type.c_str()))
      valid_class = 0;

    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if(gt[i].occlusion>MAX_OCCLUSION[difficulty] || gt[i].truncation>MAX_TRUNCATION[difficulty] || height<MIN_HEIGHT[difficulty])
      ignore = true;

    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;
    else
      valid_class = -1;

    int32_t height = fabs(det[i].box.y1 - det[i].box.y2);

    // set ignored vector for detections
    if(height<MIN_HEIGHT[difficulty])
      ignored_det.push_back(1);
    else if(valid_class==1)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

tPrData computeStatistics(CLASSES current_class, const vector<tGroundtruth> &gt,
        const vector<tDetection> &det, const vector<tGroundtruth> &dc,
        const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det,
        bool compute_fp, double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        METRIC metric, bool compute_aos=false, double thresh=0, bool debug=false){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j], gt[i], -1);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>MIN_OVERLAP[metric][current_class] && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>MIN_OVERLAP[metric][current_class] && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>MIN_OVERLAP[metric][current_class] && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0) {
      stat.fn++;
    }

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // if FP are requested, consider stuff area
  if(compute_fp){

    // count fp
    for(int32_t i=0; i<det.size(); i++){

      // count false positives if required (height smaller than required is ignored (ignored_det==1)
      if(!(assigned_detection[i] || ignored_det[i]==-1 || ignored_det[i]==1 || ignored_threshold[i]))
        stat.fp++;
    }

    // do not consider detections overlapping with stuff area
    int32_t nstuff = 0;
    for(int32_t i=0; i<dc.size(); i++){
      for(int32_t j=0; j<det.size(); j++){

        // detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
        if(assigned_detection[j])
          continue;
        if(ignored_det[j]==-1 || ignored_det[j]==1)
          continue;
        if(ignored_threshold[j])
          continue;

        // compute overlap and assign to stuff area, if overlap exceeds class specific value
        double overlap = boxoverlap(det[j], dc[i], 0);
        if(overlap>MIN_OVERLAP[metric][current_class]){
          assigned_detection[j] = true;
          nstuff++;
        }
      }
    }

    // FP = no. of all not to ground truth assigned detections - detections assigned to stuff areas
    stat.fp -= nstuff;

    // if all orientation values are valid, the AOS is computed
    if(compute_aos){
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);

      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }
  }
  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori, CLASSES current_class,
        const vector< vector<tGroundtruth> > &groundtruth,
        const vector< vector<tDetection> > &detections, bool compute_aos,
        double (*boxoverlap)(tDetection, tGroundtruth, int32_t),
        vector<double> &precision, vector<double> &aos,
        DIFFICULTY difficulty, METRIC metric) {
    assert(groundtruth.size() == detections.size());

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  // for all test images do
  for (int32_t i=0; i<groundtruth.size(); i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false, boxoverlap, metric);

    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  // get scores that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);

  // compute TP,FP,FN for relevant scores
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());
  for (int32_t i=0; i<groundtruth.size(); i++){

    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, boxoverlap, metric,
                              compute_aos, thresholds[t], t==38);

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1)
        pr[t].similarity += tmp.similarity;
    }
  }

  // compute recall, precision and AOS
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if(compute_aos)
    aos.assign(N_SAMPLE_PTS, 0);
  double r=0;
  for (int32_t i=0; i<thresholds.size(); i++){
    r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);
    if(compute_aos)
      aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
  }

  // filter precision and AOS using max_{i..end}(precision)
  for (int32_t i=0; i<thresholds.size(); i++){
    precision[i] = *max_element(precision.begin()+i, precision.end());
    if(compute_aos)
      aos[i] = *max_element(aos.begin()+i, aos.end());
  }

  // save statisics and finish with success
  saveStats(precision, aos, fp_det, fp_ori);
    return true;
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  printf("save %s\n", (dir_name + "/" + file_name + ".txt").c_str());
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i]);
  fclose(fp);

  float sum[3] = {0, 0, 0};
  for (int v = 0; v < 3; ++v)
      for (int i = 0; i < vals[v].size(); i = i + 4)
          sum[v] += vals[v][i];
  printf("%s AP: %f %f %f\n", file_name.c_str(), sum[0] / 11 * 100, sum[1] / 11 * 100, sum[2] / 11 * 100);


  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

vector<int32_t> getEvalIndices(const string& result_dir) {

    DIR* dir;
    dirent* entity;
    dir = opendir(result_dir.c_str());
    if (dir) {
        while (entity = readdir(dir)) {
            string path(entity->d_name);
            int32_t len = path.size();
            if (len < 10) continue;
            int32_t index = atoi(path.substr(len - 10, 10).c_str());
            indices.push_back(index);
        }
    }
    return indices;
}

bool eval(string gt_dir, string result_dir, Mail* mail){

  // set some global parameters
  initGlobals();

  // ground truth and result directories
  // string gt_dir         = "data/object/label_2";
  // string result_dir     = "results/" + result_sha;
  string plot_dir       = result_dir + "/plot";

  // create output directories
  system(("mkdir " + plot_dir).c_str());

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true;
  vector<bool> eval_image(NUM_CLASS, false);
  vector<bool> eval_ground(NUM_CLASS, false);
  vector<bool> eval_3d(NUM_CLASS, false);

  // for all images read groundtruth and detections
  mail->msg("Loading detections...");
  std::vector<int32_t> indices = getEvalIndices(result_dir + "/data/");
  printf("number of files for evaluation: %d\n", (int)indices.size());

  for (int32_t i=0; i<indices.size(); i++) {

    // file name
    char file_name[256];
    sprintf(file_name,"%06d.txt",indices.at(i));

    // read ground truth and result poses
    bool gt_success,det_success;
    vector<tGroundtruth> gt   = loadGroundtruth(gt_dir + "/" + file_name,gt_success);
    // std::cout<<"file_name is "<<file_name<<std::endl;
    // for (int32_t i=0; i<gt.size(); i++)
    //   std::cout<<gt[i].box.type<<"   "<< gt[i].truncation<<"   "<<gt[i].occlusion<<"   "<< gt[i].box.alpha<<"   "<<
    //   gt[i].box.x1<<"   "<< gt[i].box.y1<<"   "<<gt[i].box.x2<<"   "<<gt[i].box.y2<<"   "<<
    //   gt[i].h<<"   "<< gt[i].w<<"   "<<gt[i].l<<"   "<<gt[i].t1<<"   "<<
    //   gt[i].t2<<"   "<< gt[i].t3<<"   "<<gt[i].ry<<std::endl;
    // exit(0);
    vector<tDetection>   det  = loadDetections(result_dir + "/data/" + file_name,
            compute_aos, eval_image, eval_ground, eval_3d, det_success);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      mail->msg("ERROR: Couldn't read: %s of ground truth. Please write me an email!", file_name);
      return false;
    }
    if (!det_success) {
      mail->msg("ERROR: Couldn't read: %s", file_name);
      return false;
    }
  }
  mail->msg("  done.");

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0;

  // eval image 2D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_image[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection.txt").c_str(), "w");
      if(compute_aos)
        fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_orientation.txt").c_str(),"w");
      vector<double> precision[3], aos[3];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[0], aos[0], EASY, IMAGE)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[1], aos[1], MODERATE, IMAGE)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, imageBoxOverlap, precision[2], aos[2], HARD, IMAGE)) {
        mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection", CLASS_NAMES[c], precision, 0);
      if(compute_aos){
        saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_orientation", CLASS_NAMES[c], aos, 1);
        fclose(fp_ori);
      }
    }
  }

  // don't evaluate AOS for birdview boxes and 3D boxes
  compute_aos = false;

  // eval bird's eye view bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_ground[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_ground.txt").c_str(), "w");
      vector<double> precision[3], aos[3];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[0], aos[0], EASY, GROUND)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[1], aos[1], MODERATE, GROUND)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, groundBoxOverlap, precision[2], aos[2], HARD, GROUND)) {
        mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_ground", CLASS_NAMES[c], precision, 0);
    }
  }

  // eval 3D bounding boxes
  for (int c = 0; c < NUM_CLASS; c++) {
    CLASSES cls = (CLASSES)c;
    if (eval_3d[c]) {
      fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[c] + "_detection_3d.txt").c_str(), "w");
      vector<double> precision[3], aos[3];
      if(   !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[0], aos[0], EASY, BOX3D)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[1], aos[1], MODERATE, BOX3D)
         || !eval_class(fp_det, fp_ori, cls, groundtruth, detections, compute_aos, box3DOverlap, precision[2], aos[2], HARD, BOX3D)) {
        mail->msg("%s evaluation failed.", CLASS_NAMES[c].c_str());
        return false;
      }
      fclose(fp_det);
      saveAndPlotPlots(plot_dir, CLASS_NAMES[c] + "_detection_3d", CLASS_NAMES[c], precision, 0);
    }
  }

  // success
  return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=3) {
    cout << "Usage: ./eval_detection_3d_offline gt_dir result_dir" << endl;
    return 1;
  }

  // read arguments
  string gt_dir = argv[1];
  string result_dir = argv[2];

  // init notification mail
  Mail *mail;
  mail = new Mail();
  mail->msg("Thank you for participating in our evaluation!");

  // run evaluation
  if (eval(gt_dir, result_dir, mail)) {
    mail->msg("Your evaluation results are available at:");
    mail->msg(result_dir.c_str());
  } else {
    system(("rm -r " + result_dir + "/plot").c_str());
    mail->msg("An error occured while processing your results.");
  }

  // send mail and exit
  delete mail;

  return 0;
}
