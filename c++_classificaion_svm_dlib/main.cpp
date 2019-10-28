#include <iostream>
#include <fstream>
#include <algorithm>    // std::swap
#include <vector>       // std::vector
#include <cstdlib>
#include <windows.h>

#include <dlib/svm_threaded.h>
#include <dlib/rand.h>
#include <dlib/matrix.h>

using namespace std;
using namespace dlib;


template <typename T>
void remove(std::vector<T>& vec, size_t pos)
{
	std::vector<T>::iterator it = vec.begin();
	std::advance(it, pos);
	vec.erase(it);
}

double calculateaccuracy(matrix<double>  confusionmatrix)
{
	int TValue = 0;
	int FValue = 0;
	for (int i = 0; i < 11; i++) {
		for (int j = 0; j < 11; j++)
		{
			if (i == j)
			{
				TValue += confusionmatrix(i, j);
			}
			else
			{
				FValue += confusionmatrix(i, j);
			}

		}

	}
	printf("T: %d, F :%d", TValue, FValue);
	return (TValue / double(TValue + FValue));
}


/*double dlib_svm_Multi_Leaveoneout(std::vector<std::vector<double>> features_, std::vector<double> label_, int feature_size_, int noSamples, double alfa)
{
	typedef matrix<double, 703, 1> sample_type; // 4 = Feature Size, cannot be set to variable !!!!

	std::vector<sample_type> feature_dlib;
	std::vector<sample_type> feature_train_dlib;
	std::vector<sample_type> feature_test_dlib;

	std::vector<double> train_label;
	std::vector<double> test_label;

	//Filling Vector_Matrix
	double TotAccuracy = 0;	//# true positives
	double accuracy = 0;

	//printf("started.. \n");
	for (int m = 0; m < noSamples; m++)
	{
		sample_type temp;
		temp.set_size(feature_size_);
		//cout << "Set size " << temp.size  << endl;

		for (int j = 0; j < feature_size_; j++)
			temp(j, 0) = features_[m][j];

		feature_dlib.push_back(temp);

	}

	clock_t begin = clock();

	//Leave-One-out
	for (int testid = 0; testid < noSamples; testid++)
	{

		double  error = 0;
		accuracy = 0;
		//printf("Feature dlib %d", feature_dlib.data()[testid]);
		feature_test_dlib.push_back(feature_dlib.data()[testid]);
		test_label.push_back(label_.data()[testid]);
		feature_train_dlib = feature_dlib;
		remove(feature_train_dlib, testid);
		train_label = label_;
		remove(train_label, testid);

		typedef linear_kernel<sample_type> linear_kernel;

		typedef svm_multiclass_linear_trainer <linear_kernel, double> svm_mc_trainer;
		svm_mc_trainer trainer;
		//printf("Thread %d \n", trainer.get_num_threads());


		multiclass_linear_decision_function<linear_kernel, double> df = trainer.train(feature_train_dlib, train_label);

		//randomize_samples(feature_dlib, label_);
		//accuracy = cross_validate_multiclass_trainer(trainer, feature_dlib, label_, 5);
		//printf("accuracy %5.1f", accuracy);
		double prediction = df(feature_test_dlib[0]);
		printf("Prediction %d, Expecct %d \n",prediction,test_label[0]);
		if (prediction != test_label[0])
		{
			error = 1;
			//printf("ERROR \n");
		}
		accuracy = 1 - (error);
		TotAccuracy += accuracy;
	}
	clock_t end = clock();
	printf("has completed in %4.1f seconds and accuracy %f \n", double(end - begin) / CLOCKS_PER_SEC, TotAccuracy);
	return TotAccuracy / noSamples;

}

*/


double dlib_svm_multiclass_kfold(std::vector<std::vector<double>> features_, std::vector<double> label_, int feature_size_, int noSamples, double alfa)
{
	typedef matrix<double, 703, 1> sample_type; // 4 = Feature Size, cannot be set to variable !!!!

	std::vector<sample_type> feature_dlib;
	double accuracy = 0;

	printf("started.. \n");
	for (int m = 0; m < noSamples; m++)
	{
		sample_type temp;
		temp.set_size(feature_size_);
		//cout << "Set size " << temp.size  << endl;

		for (int j = 0; j < feature_size_; j++)
			temp(j, 0) = features_[m][j];

		feature_dlib.push_back(temp);

	}

	clock_t begin = clock();

	typedef linear_kernel<sample_type> linear_kernel;

	typedef svm_multiclass_linear_trainer <linear_kernel, double> svm_mc_trainer;
	svm_mc_trainer trainer;

	//multiclass_linear_decision_function<linear_kernel,double> df = trainer.train(feature_dlib, label_); 


	matrix<double, 11, 11>  confusionmatrix;
	confusionmatrix.set_size(11, 11);
	confusionmatrix = cross_validate_multiclass_trainer(trainer, feature_dlib, label_, 5);
	accuracy = calculateaccuracy(confusionmatrix);

	clock_t end = clock();
	printf("has completed in %4.1f seconds and accuracy %f\n", double(end - begin) / CLOCKS_PER_SEC, accuracy);
	return accuracy;

}


double dlib_svm_kfold(std::vector<std::vector<double>> features_, std::vector<double> label_, int feature_size_, int noSamples, double alfa)
{
	typedef matrix<double, 703, 1> sample_type; // 4 = Feature Size, cannot be set to variable !!!!

	std::vector<sample_type> feature_dlib;
	double accuracy = 0;

	printf("started.. \n");
	for (int m = 0; m < noSamples; m++)
	{
		sample_type temp;
		temp.set_size(feature_size_);
		//cout << "Set size " << temp.size  << endl;

		for (int j = 0; j < feature_size_; j++)
			temp(j, 0) = features_[m][j];

		feature_dlib.push_back(temp);

	}

	clock_t begin = clock();
	
/*	typedef linear_kernel<sample_type> linear_kernel; 

	typedef svm_multiclass_linear_trainer <linear_kernel, double> svm_mc_trainer;
	svm_mc_trainer trainer;
	

	multiclass_linear_decision_function<linear_kernel,double> df = trainer.train(feature_dlib, label_); */

	typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;

	ovo_trainer trainer;
	typedef linear_kernel<sample_type> linear_kernel;

	krr_trainer<linear_kernel> linear_trainer;
	
	linear_trainer.set_kernel(linear_kernel());
	trainer.set_trainer(linear_trainer);        // linear_trainer
	randomize_samples(feature_dlib, label_);
   //printf("Training start \n");
	//one_vs_one_decision_function<ovo_trainer> df = trainer.train(feature_dlib, label_);

	matrix<double, 11,11>  confusionmatrix;
	confusionmatrix.set_size(11, 11);
	confusionmatrix=cross_validate_multiclass_trainer(trainer, feature_dlib, label_, 5);
	accuracy= calculateaccuracy(confusionmatrix);
	
	clock_t end = clock();
	printf("has completed in %4.1f seconds and accuracy %f\n", double(end - begin) / CLOCKS_PER_SEC,accuracy);
	return accuracy;

}
double dlib_svm(std::vector<std::vector<double>> features_, std::vector<double> label_, int feature_size_, int noSamples, double alfa)
{
	typedef matrix<double, 703, 1> sample_type; // 4 = Feature Size, cannot be set to variable !!!!

	std::vector<sample_type> feature_dlib;

	std::vector<sample_type> feature_train_dlib;
	std::vector<sample_type> feature_test_dlib;

	std::vector<double> train_label;
	std::vector<double> test_label;

	//Filling Vector_Matrix
	double TotAccuracy = 0;	//# true positives
	double accuracy = 0;
	
	printf("started.. \n");
	for (int m = 0; m < noSamples; m++)
	{
		sample_type temp;
		temp.set_size(feature_size_);
		//cout << "Set size " << temp.size  << endl;

		for (int j = 0; j < feature_size_; j++)
			temp(j, 0) = features_[m][j];
		
		feature_dlib.push_back(temp);
	
	}

	clock_t begin = clock();

	//Leave-One-out
	for (int testid = 0; testid < noSamples; testid++)
	{
		
		double  error = 0;
		accuracy = 0;
		//printf("Feature dlib %d", feature_dlib.data()[testid]);
		feature_test_dlib.push_back(feature_dlib.data()[testid]);
		test_label.push_back(label_.data()[testid]);
		feature_train_dlib = feature_dlib;
		remove(feature_train_dlib, testid);
		train_label = label_;
		remove(train_label, testid);

	/*	for (int m = 0; m < noSamples; m++)
		{
				sample_type temp;
				for (int j = 0; j < feature_size_; j++)
					temp(j, 0) = features_[m][j];
				if (m == testid) {
					feature_test_dlib.push_back(temp);
					test_label.push_back(label_[m]);
				}
				else
				{
					feature_train_dlib.push_back(temp);
					train_label.push_back(label_[m]);
				}
			
		}*/

	
		
		typedef one_vs_one_trainer<any_trainer<sample_type> > ovo_trainer;

		ovo_trainer trainer;
		//typedef radial_basis_kernel<sample_type> rbf_kernel;
		typedef linear_kernel<sample_type> linear_kernel; 
		//krr_trainer<rbf_kernel> rbf_trainer;

		//svm_nu_trainer <linear_kernel> linear_trainer;


		krr_trainer<linear_kernel> linear_trainer;
		//svm_multiclass_linear_trainer <rbf_kernel> rbf_trainer;
		int prediction, test_gt;
		

		//rbf_trainer.set_kernel(rbf_kernel(alfa));
		linear_trainer.set_kernel(linear_kernel());
		//trainer.set_trainer(rbf_trainer);        // rbf_trainer
		 trainer.set_trainer(linear_trainer);        // linear_trainer
		 trainer.set_num_threads(7);
		//randomize_samples(feature_train_dlib, train_label);
		//printf("Training start \n");
		one_vs_one_decision_function<ovo_trainer> df = trainer.train(feature_train_dlib, train_label);

	
		//std::cout << "Testing started" << endl;
		prediction = df(feature_test_dlib[0]);
		test_gt = test_label[0];
	
		if (prediction != test_gt) error=1;

		accuracy = 1 - (error);
		//std::cout << "Local accuracy: " << accuracy << endl;

		feature_test_dlib.clear();
		feature_train_dlib.clear();
		test_label.clear();
		train_label.clear();

		
		TotAccuracy += accuracy;

	}
	clock_t end = clock();
	printf("has completed in %4.1f seconds\n", double(end - begin) / CLOCKS_PER_SEC);
	
	//printf("One Leave out accuracy %f",(TotAccuracy/ noSamples));
	/*for (int i = 0; i < train_features_.size()-100; i++)
	{
		sample_type temp;
		for (int j = 0; j <feature_size_; j++)
			temp(j, 0) = train_features_[i][j];
		feature_train_dlib.push_back(temp);
		train_label_1.push_back(train_label_[i]);
	}

	for (int i = test_features_.size()-100; i < test_features_.size(); i++)
	{
		sample_type temp;
		for (int j = 0; j <feature_size_; j++)
			temp(j, 0) = test_features_[i][j];
		feature_test_dlib.push_back(temp);
		test_label_1.push_back(test_label_[i]);
	}*/

	

	//cout << "feature_train_dlib.size: " << feature_train_dlib.size() << endl;
	//cout << "feature_train_dlib[0].size: " << feature_train_dlib[0].size() << endl;
	//cout << "\nfeature_test_dlib.size: " << feature_test_dlib.size() << endl;
	//cout << "feature_test_dlib[0].size: " << feature_test_dlib[0].size() << endl;
	//cout << "\ntrain labels size: " << train_label_1.size() << endl;
	//cout << "test labels size: " << test_label_1.size() << endl;

	feature_dlib.clear();
	return (TotAccuracy / noSamples);

}


int main()
{
	string resultsFilename = "Results_svm.txt";
	string dataFilename = "Pathogen_VOC_Dataset_v4.txt";
	FILE *outFile = fopen(resultsFilename.c_str(), "a+");	//reset file
	fclose(outFile);

	int noClasses = 12;
	int noSamples = 336;
	int noFeatures = 703;
	std::vector<string> classLabels;
	std::vector<double> classIDs;
	std::vector<std::vector<double>> data;
	//open & read data file
	FILE *dataFile = fopen(dataFilename.c_str(), "r");

	int tmpInt1, tmpInt2;
	string tmpStr;
	char* buffer = new char[200];

	fscanf(dataFile, "%d", &noClasses);	//# of classes
	fgets(buffer, 100, dataFile);		//dummy read till read EOL


	classLabels.push_back("");	//dummy push to start index from 1 for class labels
	for (int i = 1; i <= noClasses; i++)
	{
		fscanf(dataFile, "%d  %s", &tmpInt1, buffer);
		classLabels.push_back(string(buffer));
	}

	fscanf(dataFile, "%d", &noSamples);		//reads # of classes from the file
	fgets(buffer, 100, dataFile);			//dummy read till read EOL

	fscanf(dataFile, "%d", &noFeatures);	//reads the feature dimension from the file
	fgets(buffer, 100, dataFile);			//dummy read till read EOL

	for (int i = 0; i < noSamples; i++)
	{
		fscanf(dataFile, "%d %d", &tmpInt1, &tmpInt2);	//read sample ID (dummy read) & class ID of the current feature vector
		classIDs.push_back(tmpInt2);	//store class ID

		data.push_back(std::vector<double>());
		for (int j = 0; j < noFeatures; j++)
		{
			fscanf(dataFile, "%d", &tmpInt1);	//read feature vector
			data[i].push_back(tmpInt1);
		}
	}


	delete buffer;
	double alfa = 0.1;  // For SVM, standart range 0-1

	
	std::vector<int> selFeatIDs;
	double Allbest = 0.0;

	//select all features in a circular manner
	for (int j = 0; j < noFeatures; j++) 
		selFeatIDs.push_back(j);

	for (int j = noFeatures; j >= 0; j--)
	{
		int eliminated = 0;
		if (j < noFeatures)	//no elimination in the first loop
		{
			eliminated = selFeatIDs[j];
			selFeatIDs.erase(selFeatIDs.begin() + j);	//jth feature is appended
			noFeatures -= 1;
		}

		std::vector<std::vector<double>> filtereddata;
	

		
		for (int m = 0; m < noSamples; m++)
		{
			filtereddata.push_back(std::vector<double>());
			for (int p = 0; p < selFeatIDs.size(); p++)
			{
				filtereddata[m].push_back(data[m][selFeatIDs[p]]);
				
			}
			//printf("Filtered Feature size %d \n", (filtereddata[m].size()));
		}
		// SVM Classifer
		//printf("Feature removal :%d", j);
		
		double accuracy = dlib_svm(filtereddata, classIDs, noFeatures, noSamples, alfa);
		// double accuracy =  dlib_svm_kfold(filtereddata, classIDs, noFeatures, noSamples, alfa);
		// double accuracy = dlib_svm_multiclass_kfold(filtereddata, classIDs, noFeatures, noSamples, alfa);
		//double accuracy =  dlib_svm_Multi_Leaveoneout(filtereddata, classIDs, noFeatures, noSamples, alfa);
		//cout << "Accuracy:\n" << endl << accuracy << endl;
		if (accuracy < Allbest)
		{
			//printf("Acuracy reduced");
			selFeatIDs.insert(selFeatIDs.begin() + j, eliminated);
			noFeatures += 1;
		}
		else
		{
			//printf("Acuracy improved or same");
			Allbest = accuracy;
		}
		printf("Accuracy :%5.3f, Overall accuracy :%5.3f and feature removal round: %d", accuracy,Allbest, (703-j));
		outFile = fopen(resultsFilename.c_str(), "a+");	//reset file

		fprintf(outFile, "%d /703 is done ;No. Selected Features: %d \n", j , selFeatIDs.size());

		fprintf(outFile, "#####  Accuracy: %5.3f ##### Best Accuracy: %5.3f #####\nSelected Feature IDs:", accuracy, Allbest);

		for (int j = 0; j < selFeatIDs.size(); j++) fprintf(outFile, "%d, ", selFeatIDs[j]);

		fprintf(outFile, "\n\n");

		fclose(outFile);
	}
	
	system("PAUSE");
	return 0;
}






