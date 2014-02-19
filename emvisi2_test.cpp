#include "emvisi2.h"
#include <opencv2/highgui.hpp>

#include <sys/time.h>

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
	cerr << "usage: " << str << " [-v] <background> <input frame> [<mask>]\n";
	cerr << "	use -v for more verbosity and more intermediate images saved.\n";
	exit(-1);
}


int main(int argc, char *argv[])
{
	EMVisi2 emv;
	int nim=0;
	cv::Mat im[3];

	// parse command line
	for (int narg=1; narg<argc; ++narg) {
		if (strcmp(argv[narg],"-v")==0) {
			emv.save_images=true;
		} else {
			im[nim] = cv::imread(argv[narg], (nim==2 ? 0 : -1));
			if (im[nim].empty()) {
				cerr << argv[narg] << ": can't load image.\n";
				exit(-2);
			}
			nim++;
		}
	}

	cv::Mat im1 = im[0];
	cv::Mat im2 = im[1];
	cv::Mat mask = im[2];

	if (im1.empty() || im2.empty()) usage(argv[0]);
	
	if ((im1.channels() != im2.channels()) || (im1.cols != im2.cols) ) {
		cerr << "image format or size do not match.\n";
		exit(-4);
	}
	int h = (im1.rows < im2.rows ? im1.rows : im2.rows);
	im1.rows=h;
	im2.rows=h;

	Timer timer;
	cout << "Initialization.. ";

	// EMVisi2 setup
	if (!emv.init()) {
		cerr << "EMVisi2::init() failed.\n";
		return -1;
	}
	emv.setModel(im1, mask);

	cout << "done in " << timer.duration() << " ms.\n";

	cout << "setTarget... ";
	timer.start();

	emv.setTarget(im2);

	cout << "computed in " << timer.duration() << " ms.\n";

	Timer iterations;

	const int niter=16;
	emv.run(niter, 0);

	float it_duration = iterations.duration();

#ifdef WITH_GRAPHCUT
	emv.smooth(2.4, 0.001);
	float gc_duration = iterations.duration() - it_duration;
#endif

	cout << niter << " iterations computed in " << it_duration << " ms. (avg " 
		<< it_duration / (float)niter << " ms per iteration)\n";
#ifdef WITH_GRAPHCUT
	cout << "graph cut computed in " << gc_duration << " ms.\n";
#endif

	float duration = timer.duration();
	cout << "       frame computed in " << duration << " ms ("
		<< (im1.cols*im1.rows)/duration << " K pix/sec).\n";

	save_proba("final_proba.png",emv.proba);
	return 0;
}

