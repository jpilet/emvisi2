#include "emvisi2.h"
#include <opencv2/highgui/highgui.hpp>

#include <sys/time.h>
#include <list>
#include <string>

using namespace cv;
using namespace std;

class Timer {
public:
	Timer() { start(); }
	void start() { gettimeofday(&s,0); }
	double duration();
private:
	struct timeval s;
};

double Timer::duration() {
	struct timeval end, dt;
	gettimeofday(&end,0);
	timersub(&end,&s, &dt);
	return dt.tv_sec * 1000.0 + dt.tv_usec/1000.0;
}

void usage(char *str)
{
	cerr << "usage: " << str << " [-v] -b <background> <input frames>\n";
	cerr << "	use -v for more verbosity and more intermediate images saved.\n";
	exit(-1);
}

string destinationFilename(const string &a) {
  return a + "_result.png";
}

int main(int argc, char *argv[])
{
  EMVisi2 emv;
  int nim=0;
  cv::Mat background;
  list<string> images;

  // parse command line
  for (int narg=1; narg<argc; ++narg) {
    if (narg + 1 < argc) {
      if (strcmp(argv[narg], "-b") == 0) {
        background = imread(argv[narg+1], -1);
        if (background.empty()) {
          cerr << argv[narg+1] << ": can't load background.\n";
          return 1;
        }
        narg++;
        continue;
      }
    }

    if (strcmp(argv[narg],"-v")==0) {
      emv.save_images=true;
    } else if (argv[narg][0] == '-') {
      usage(argv[0]);
    } else {
      images.push_back(argv[narg]);
    }
  }

  if (background.empty()) {
    usage(argv[0]);
    return 2;
  }

  Timer timer;
  cout << "Initialization.. ";

  // EMVisi2 setup
  if (!emv.init()) {
    cerr << "EMVisi2::init() failed.\n";
    return -1;
  }
  emv.setModel(background);

  cout << "done in " << timer.duration() << " ms.\n";

  for (list<string>::const_iterator it = images.begin();
       it != images.end(); ++it) {
    Mat frame = imread(*it, -1);
    if (frame.empty()) {
      cerr << *it << ": can't load frame, skipping.\n";
      continue;
    }

    cout << *it << ": ";
    timer.start();

    emv.setTarget(frame);
    emv.run(5, 0);

    cout << "computed in " << timer.duration() << " ms.\n";

    save_proba(destinationFilename(*it).c_str(),emv.proba);
  }
  return 0;
}

