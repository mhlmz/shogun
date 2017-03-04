/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 */

#include <gtest/gtest.h>
#include <shogun/lib/config.h>
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/machine/LinearMachine.h>
#include <shogun/regression/LinearRidgeRegression.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/base/class_list.h>
#include "utils/Utils.h"

using namespace shogun;

#ifdef HAVE_LAPACK

TEST(MachineSerialization, linear_ridge_regression)
{
    /* generate train and test data */
	index_t n_train=30;
	index_t n_test=10;
	float64_t a=3.;

	SGMatrix<float64_t> X(1, n_train);
	SGMatrix<float64_t> X_test(1, n_test);
	SGVector<float64_t> Y(n_train);

	for (index_t i=0; i<n_train; ++i)
	{
		X(0,i)=i;
		Y[i]=a*X(0,i);
	}

	for (index_t i=0; i<n_test; ++i)
		X_test(0,i)=i+0.5;

	CDenseFeatures<float64_t>* feat_train=new CDenseFeatures<float64_t>(X);
	CDenseFeatures<float64_t>* feat_test=new CDenseFeatures<float64_t>(X_test);
	CRegressionLabels* label_train=new CRegressionLabels(Y);

	/* instantiate and train the machine */
	float64_t tau=1e-3;
	CLinearRidgeRegression* model=new CLinearRidgeRegression(tau, feat_train, label_train);
	bool train_success = model->train();
	ASSERT_TRUE(train_success);

	/* predict with trained model */
	CRegressionLabels* predictions=model->apply_regression(feat_test);
	SGVector<float64_t> prediction_vector=predictions->get_labels();

	/* serialize the model */
	std::string class_name("LinearRidgeRegression");
	std::string file_template = "/tmp/shogun-unittest-serialization-ascii-" + class_name + ".XXXXXX";
	char* filename = mktemp_cst(const_cast<char*>(file_template.c_str()));

	CSerializableAsciiFile *file=new CSerializableAsciiFile(filename, 'w');
	bool save_success = model->save_serializable(file);
	file->close();
	SG_UNREF(file);
	ASSERT_TRUE(save_success);

	/* deserialize the model */
	file=new CSerializableAsciiFile(filename, 'r');
	CLinearRidgeRegression* deserialized_model = (CLinearRidgeRegression*)new_sgserializable(class_name.c_str(), PT_NOT_GENERIC);
	ASSERT_TRUE(deserialized_model != NULL);
	bool load_success = deserialized_model->load_serializable(file);
	file->close();
	SG_UNREF(file);
	ASSERT_TRUE(load_success);

	/* predict with deserialized model */
	CRegressionLabels* deserialized_predictions=deserialized_model->apply_regression(feat_test);
	SGVector<float64_t> deserialized_prediction_vector=deserialized_predictions->get_labels();

	/* check whether the predictions are equal */
	for (index_t i=0; i<n_test; ++i)
		EXPECT_DOUBLE_EQ(prediction_vector[i], deserialized_prediction_vector[i]);

	int delete_success = unlink(filename);
	ASSERT_EQ(0, delete_success);
	
	SG_UNREF(model);
	SG_UNREF(deserialized_model);
	SG_UNREF(predictions);
	SG_UNREF(deserialized_predictions);
}

#endif /* HAVE_LAPACK */
