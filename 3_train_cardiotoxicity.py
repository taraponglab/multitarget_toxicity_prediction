import pandas as pd
import numpy as np
import os
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, roc_auc_score, balanced_accuracy_score, roc_curve, matthews_corrcoef, precision_score, precision_recall_curve, auc, average_precision_score
from sklearn.model_selection import cross_val_predict, StratifiedKFold, KFold, LeaveOneOut
import xgboost as xgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import GridSearchCV
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import joblib
from imblearn.over_sampling import ADASYN

def remove_constant_string_des(df):
    #delete string value
    df = df.select_dtypes(exclude=['object'])
    #delete constant value
    for column in df.columns:
        if df[column].nunique() == 1:  # This checks if the column has only one unique value
            df = df.drop(column, axis=1)  # This drops the column from the DataFrame
    return df

def remove_highly_correlated_features(df, threshold=0.7):
    # Compute pairwise correlation of columns
    corr_matrix = df.corr().abs()
    # Create a mask for the upper triangle
    upper = corr_matrix.where(
        pd.DataFrame(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool), 
                     index=corr_matrix.index, columns=corr_matrix.columns)
    )
    # Identify columns to drop based on threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    # Drop the columns from the DataFrame
    df_dropped = df.drop(columns=to_drop)
    return df_dropped

def y_prediction(model, x_train, y_train, col_name):
    y_pred = pd.DataFrame(model.predict(x_train), columns=[col_name]).set_index(y_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)
    mcc = matthews_corrcoef(y_train, y_pred)
    auroc = roc_auc_score(y_train, y_pred)
    auprc = average_precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auroc],
        'AUPRC': [auprc],
    }, index=[col_name])
    return y_pred, metrics

def y_prediction_cv(model, x_train, y_train, col_name):
    y_pred = cross_val_predict(model, x_train, y_train, cv=5)
    y_pred = pd.DataFrame(y_pred, columns=[col_name]).set_index(y_train.index)
    acc = accuracy_score(y_train, y_pred)
    sen = recall_score(y_train, y_pred)
    mcc = matthews_corrcoef(y_train, y_pred)
    auroc = roc_auc_score(y_train, y_pred)
    auprc = average_precision_score(y_train, y_pred)
    # Calculate specificity
    tn, fp, fn, tp = confusion_matrix(y_train, y_pred).ravel()
    spc = tn / (tn + fp)

    # Create a DataFrame to store the metrics
    metrics = pd.DataFrame({
        'Accuracy': [acc],
        'Sensitivity': [sen],
        'Specificity': [spc],
        'MCC': [mcc],
        'AUC': [auroc],
        'AUPRC': [auprc],
    }, index=[col_name])
    return y_pred, metrics


def plot_auc_auprc_cv(model, x_train, y_train, col_name):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    precisions = []
    auprcs = []
    mean_recall = np.linspace(0, 1, 100)

    # ROC Curve
    plt.figure(figsize=(4, 3))
    for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(x_tr, y_tr)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_val)[:, 1]
        else:
            y_score = model.decision_function(x_val)
        # ROC
        fpr, tpr, _ = roc_curve(y_val, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.7, label=f"Fold {i+1} (AUC = {roc_auc:.2f})")
        # Interpolate tpr
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    # Plot mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f"Mean AUROC = {mean_auc:.2f} ± {std_auc:.2f}", lw=2)
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("1-Specificity", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Sensitivity", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUROC - {col_name}", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_roc_auc_cv.png"), dpi=500)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(5, 3))
    for i, (train_idx, val_idx) in enumerate(skf.split(x_train, y_train)):
        x_tr, x_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(x_tr, y_tr)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_val)[:, 1]
        else:
            y_score = model.decision_function(x_val)
        precision, recall, _ = precision_recall_curve(y_val, y_score)
        auprc = average_precision_score(y_val, y_score)
        auprcs.append(auprc)
        plt.plot(recall, precision, lw=1, alpha=0.7, label=f"Fold {i+1} (AUPRC = {auprc:.2f})")
        # Interpolate precision
        precisions.append(np.interp(mean_recall, recall[::-1], precision[::-1]))
    # Plot mean PRC
    mean_precision = np.mean(precisions, axis=0)
    mean_auprc = np.mean(auprcs)
    std_auprc = np.std(auprcs)
    plt.plot(mean_recall, mean_precision, color='b', label=f"Mean AUPRC = {mean_auprc:.2f} ± {std_auprc:.2f}", lw=2)
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Precision", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUPRC - {col_name}", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_prc_auprc_cv.png"), dpi=500)
    plt.close()

# Plot AUROC and AUPRC for Test Set
def plot_auc_auprc_test(model, x_test, y_test, col_name):
    plt.figure(figsize=(4, 3))
    # Predict probabilities
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(x_test)[:, 1]
    else:
        y_score = model.decision_function(x_test)
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='b', lw=2, label=f"AUROC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel("1-Specificity", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Sensitivity", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUROC - {col_name} (Test Set)", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_roc_auc_test.png"), dpi=500)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(5, 3))
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    auprc = average_precision_score(y_test, y_score)
    plt.plot(recall, precision, color='b', lw=2, label=f"AUPRC = {auprc:.2f}")
    plt.xlabel("Recall (Sensitivity)", fontsize=12, fontstyle='italic', weight="bold")
    plt.ylabel("Precision", fontsize=12, fontstyle='italic', weight="bold")
    plt.title(f"AUPRC - {col_name} (Test Set)", fontsize=12, fontstyle='italic', weight="bold")
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_metrics",f"{col_name}_prc_auprc_test.png"), dpi=500)
    plt.close()
    

def plot_confusion_matrix_with_acc(y_true, y_pred, class_names, title="Confusion Matrix", filename="confusion_matrix.png"):
    """
    Plots a confusion matrix with per-class accuracy (%) in each box.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.round(cm / cm_sum.astype(float) * 100, 1)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(2 + n_classes, 2 + n_classes))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_title(title, fontsize=12, fontstyle='italic', fontweight='bold')

    # Loop over data dimensions and create text annotations.
    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(j, i, f'{value}\n{perc:.1f}%', ha="center", va="center",
                    color="white" if perc > thresh else "black", fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs("graph_metrics", exist_ok=True)
    plt.savefig(os.path.join("graph_metrics", filename), dpi=500, bbox_inches='tight')
    plt.close()

    

def plot_confusion_matrix_cv(y_true, y_pred, class_names, title="Confusion Matrix (CV)", filename="confusion_matrix_cv.png"):
    """
    Plots a confusion matrix with per-class accuracy (%) in each box for CV predictions.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_perc = np.round(cm / cm_sum.astype(float) * 100, 1)
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(2 + n_classes, 2 + n_classes))
    im = ax.imshow(cm_perc, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(n_classes))
    ax.set_yticks(np.arange(n_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    ax.set_xlabel('Predicted label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_ylabel('True label', fontsize=12, fontstyle='italic', fontweight='bold')
    ax.set_title(title, fontsize=12, fontstyle='italic', fontweight='bold')

    thresh = 50
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = cm[i, j]
            perc = cm_perc[i, j]
            ax.text(j, i, f'{value}\n{perc:.1f}%', ha="center", va="center",
                    color="white" if perc > thresh else "black", fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs("graph_metrics", exist_ok=True)
    plt.savefig(os.path.join("graph_metrics", filename), dpi=500, bbox_inches='tight')
    plt.close()


def nearest_neighbor_AD(x_train, x_test, name, k, z=3):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean').fit(x_train)
    dump(nn, os.path.join("graph_ad", "ad_"+ str(k) +"_"+ str(z) +".joblib"))
    distance, index = nn.kneighbors(x_train)
    # Calculate mean and sd of distance in train set
    di = np.mean(distance, axis=1)
    # Find mean and sd of di
    dk = np.mean(di)
    sk = np.std(di)
    print('dk = ', dk)
    print('sk = ', sk)
    # Calculate di of test
    distance, index = nn.kneighbors(x_test)
    di = np.mean(distance, axis=1)
    AD_status = ['within_AD' if di[i] < dk + (z * sk) else 'outside_AD' for i in range(len(di))]

    # Create DataFrame with index from x_test and the respective status
    df = pd.DataFrame(AD_status, index=x_test.index, columns=['AD_status'])
    return df, dk, sk

def run_ad(stacked_model, stack_train, stack_test, y_test, name, z = 0.5):
    k_values = [5,6,7, 8, 9, 10]
    ACC_values = []
    MCC_values = []
    AUROC_values = []
    AUPRC_values = []
    removed_compounds_values = []
    dk_values = []
    sk_values = []
    
    for i in k_values:
        print('k = ', i, 'z=', str(z))
        t, dk, sk = nearest_neighbor_AD(stack_train, stack_test, name, i, z=z)
        print(t['AD_status'].value_counts())
        x_ad_test = stack_test[t['AD_status'] == 'within_AD']
        y_ad_test = y_test.loc[x_ad_test.index]
        y_pred_test = stacked_model.predict(x_ad_test)
        print(len(x_ad_test),len(y_ad_test), len(y_pred_test) )
        acc = round(accuracy_score(y_ad_test, y_pred_test), 3)
        auroc = round(roc_auc_score(y_ad_test, y_pred_test), 3)
        mcc = round(matthews_corrcoef(y_ad_test, y_pred_test), 3)
        auprc = round(average_precision_score(y_ad_test, y_pred_test), 3)
        print('MCC: ', mcc,'AUROC: ', auroc,'AUPRC:', auprc)
        ACC_values.append(acc)
        MCC_values.append(mcc)
        AUROC_values.append(auroc)
        AUPRC_values.append(auprc)
        removed_compounds_values.append((t['AD_status'] == 'outside_AD').sum())
        dk_values.append(dk)
        sk_values.append(sk)
    k_values   = np.array(k_values)
    ACC_values = np.array(ACC_values)
    MCC_values = np.array(MCC_values)
    AUROC_values = np.array(AUROC_values)
    AUPRC_values = np.array(AUPRC_values)
    dk_values  = np.array(dk_values)
    sk_values  = np.array(sk_values)
    removed_compounds_values = np.array(removed_compounds_values)
    # Save table
    ad_metrics = pd.DataFrame({
        "k": k_values[:len(MCC_values)],  # Adjust if some values are skipped
        "Accuracy": ACC_values,
        "MCC": MCC_values,
        "AUROC": AUROC_values,
        "AUPRC": AUPRC_values,
        "Removed Compounds": removed_compounds_values,
        "dk_values": dk_values,
        "sk_values": sk_values
    })
    ad_metrics = round(ad_metrics, 3)
    ad_metrics.to_csv(os.path.join("graph_ad","AD_metrics_"+name+"_"+ str(z)+ ".csv"))
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))
    
    ax1.plot(k_values, ACC_values,  'bo-',  label = "Accuracy")
    ax1.plot(k_values, MCC_values, 'r^-', label = "MCC")
    ax1.plot(k_values, AUROC_values, 'md-', label = "AUROC")
    ax1.plot(k_values, AUPRC_values, 'gs-', label = "AUPRC")
    # Adding labels and title
    ax1.set_xlabel('k',      fontsize=12, fontstyle='italic',weight="bold")
    ax1.set_ylabel('Scores', fontsize=12, fontstyle='italic', weight='bold')
    ax1.set_xticks(k_values)
    ax1.legend(loc='upper left', fontsize='small', bbox_to_anchor=(1.05, 1.02))
    # Second plot: Bar plot for removed_compounds_values
    ax2.bar(k_values, removed_compounds_values, color='green', edgecolor='black', alpha=0.5, width=0.3)
    ax2.set_xlabel('k', fontsize=12, fontstyle='italic',weight="bold")
    ax2.set_ylabel('Removed compounds', fontsize=12, fontstyle='italic', weight='bold')
    ax2.set_xticks(k_values)
    plt.tight_layout()
    plt.savefig(os.path.join("graph_ad","AD_"+name+"_"+ str(z)+ "_Classification_separated.png"), bbox_inches='tight', dpi=500) 
    plt.close

def y_random(x_train, x_test, y_train, y_test, metric_train, metric_test, name):
    ACC_test = []
    ACC_train = []
    for i in range(1, 101):
        y_train_shuffled = y_train.sample(frac=1, replace=False, random_state=i)
        #model = RandomForestClassifier(max_depth=3, max_features=10, random_state=None).fit(x_train, y_train_shuffled)
        model = xgb.XGBClassifier(max_depth=None, random_state=None).fit(x_train, y_train_shuffled)
        y_pred_test = model.predict(x_test)
        y_pred_train = model.predict(x_train)
        acc_test = accuracy_score(y_test, y_pred_test)
        acc_train = accuracy_score(y_train_shuffled, y_pred_train)
        ACC_test.append(acc_test)
        ACC_train.append(acc_train)
    size = [50]
    sizes = [20]

    # save the metrics
    metrics_df = pd.DataFrame({
        'Sensitivity_Train': ACC_train,
        'Sensitivity_Test': ACC_test
    })
    metrics_df.to_csv(os.path.join("graph_yrandom", f"Y-randomization-{name}-Sensitivity_classification.csv"), index=False)

    # Use the correct index for your metrics DataFrames
    x = [metric_train.loc[name, 'Sensitivity']]
    y = [metric_test.loc[name, 'Sensitivity']]
    plt.figure(figsize=(4, 3))
    plt.axvline(0.5, c='black', ls=':')
    plt.axhline(0.5, c='black', ls=':')
    plt.scatter(x, y, s=size, c=['red'], marker='x', label='Our model')
    plt.scatter(ACC_train, ACC_test, c='blue', edgecolors='black', alpha=0.7, s=sizes, label='Y-randomization')
    plt.xlabel('Sensitivity$_{5-CV}$', fontsize=14, fontstyle='italic', weight='bold')
    plt.ylabel('Sensitivity$_{Test}$', fontsize=14, fontstyle='italic', weight='bold')
    plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join("graph_yrandom", f"Y-randomization-{name}-Sensitivity_classification.png"), bbox_inches='tight', dpi=500)
    plt.close()

def compute_and_plot_shap(model, x_train, x_test, output_prefix, top_n=20, save_dir="graph_shap"):

    os.makedirs(save_dir, exist_ok=True)

    # Create SHAP explainer
    explainer = shap.Explainer(model, x_train)
    shap_values = explainer(x_test, check_additivity=False)

    # Handle classification: shap_values.values is (samples, features, classes)
    if shap_values.values.ndim == 3:
        shap_array = shap_values.values[:, :, 1]  # Use class 1 (Active)
    else:
        shap_array = shap_values.values  # For regression or single output

    # Create DataFrame
    shap_df = pd.DataFrame(shap_array, columns=x_test.columns, index=x_test.index)

    # Save SHAP values
    shap_csv_path = os.path.join(save_dir, f"{output_prefix}_shap_values.csv")
    shap_df.to_csv(shap_csv_path)
    print(f"SHAP values saved to: {shap_csv_path}")

    # Compute mean absolute SHAP
    mean_abs_shap = np.abs(shap_df).mean().sort_values(ascending=False)

    # Plot bar chart
    plt.figure(figsize=(5, max(4, 0.3 * top_n)))
    mean_abs_shap.iloc[:top_n].plot(kind='barh', color='skyblue', edgecolor='black')
    plt.xlabel("Mean |SHAP value|", fontsize=12, fontstyle='italic', weight='bold')
    plt.title(f"Top {top_n} SHAP Feature Importances", fontsize=12, fontstyle='italic', weight='bold')
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{output_prefix}_shap_barplot.png")
    plt.savefig(plot_path, dpi=500)
    plt.close()
    print(f"SHAP bar plot saved to: {plot_path}")


def main():
    all_results = []
    for x in ['AP2DC','AP2D','EState','CDKExt','CDK','CDKGraph','KRFPC','KRFP','MACCS','PubChem','SubFPC','SubFP', 'Desc']:
        print("#"*100) 
        x_train = pd.read_csv(os.path.join("Cardiotoxicity", "train", x+".csv"), index_col=0)
        y_train = pd.read_csv(os.path.join("Cardiotoxicity", "train", "y_train.csv"), index_col=0)
        x_test  = pd.read_csv(os.path.join("Cardiotoxicity", "test",  x+".csv"), index_col=0)
        y_test  = pd.read_csv(os.path.join("Cardiotoxicity", "test",  "y_test.csv"), index_col=0)
        print("Train and test data loaded")
        # Remove constant features
        x_train = remove_constant_string_des(x_train)
        x_train = remove_highly_correlated_features(x_train, threshold=0.7)
        # Fit x_test columns to match x_train
        x_test = x_test[x_train.columns]
        print("Match columns: ",x, "for",  x_train.shape, x_test.shape) 
        # Map labels to numeric values
        label_map = {'Inactive': 0, 'Active': 1}
        y_train = y_train.replace(label_map)
        y_test = y_test.replace(label_map)
        #
        y_train = y_train.infer_objects(copy=False)
        y_test  = y_test.infer_objects(copy=False)
        
        print("y_train count active and inactive:", y_train.value_counts())
        print("y_test count active and inactive:", y_test.value_counts())
        #
        # Squeeze the data to 1D for y_train and y_test to avoid issues
        y_train = y_train.squeeze()
        y_test = y_test.squeeze()
        
        print("x_train shape:", x_train.shape)
        print("y_train shape:", y_train.shape)
        
        # Check if the index of x_train and y_train match
        if not x_train.index.equals(y_train.index):
            raise ValueError("Index of x_train and y_train do not match.")
        # Check if the index of x_test and y_test match
        if not x_test.index.equals(y_test.index):
            raise ValueError("Index of x_test and y_test do not match.")
        
        # Scale the data
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled  = scaler.transform(x_test)
        # Convert to DataFrame
        x_train_scaled = pd.DataFrame(x_train_scaled, index=x_train.index, columns=x_train.columns)
        x_test_scaled  = pd.DataFrame(x_test_scaled,  index=x_test.index,  columns=x_test.columns)
        
        print("x_train_scaled shape:", x_train_scaled.shape)
        print("y_train shape:", y_train.shape)

        sm = ADASYN(random_state=42, sampling_strategy='auto')
        x_train_scaled, y_train = sm.fit_resample(x_train_scaled, y_train) 

        print("resampling x_train_scaled shape:", x_train_scaled.shape)
        print("resampling y_train shape:", y_train.shape)
        
        # Train the model
        #model = RandomForestClassifier(max_depth=3, max_features=10, random_state=None).fit(x_train_scaled, y_train)
        model = xgb.XGBClassifier(max_depth=None, random_state=None).fit(x_train_scaled, y_train)
       
        compute_and_plot_shap(model, x_train_scaled, x_test_scaled, output_prefix=f"rf_{x}")
        
        # Predict 5-cv and test
        y_pred_cv, metrics_cv = y_prediction_cv(model, x_train_scaled, y_train, x)
        y_pred_test, metrics_test = y_prediction(model, x_test_scaled, y_test, x)
        # Plot ROC and PRC
        plot_auc_auprc_cv(model, x_train_scaled, y_train, x)
        plot_auc_auprc_test(model, x_test_scaled, y_test, x)
        # Plot confusion matrix
        plot_confusion_matrix_with_acc(y_test, y_pred_test, ["Non-toxic", "Toxic"], title="Confusion Matrix - Test Set", filename=f"confusion_matrix_{x}_test.png")
        plot_confusion_matrix_cv(y_train, y_pred_cv, ["Non-toxic", "Toxic"], title="Confusion Matrix - CV", filename=f"confusion_matrix_{x}_cv.png")
        # AD analysis
        run_ad(model, x_train_scaled, x_test_scaled, y_test, x, z=0.5)
        # Y-randomization
        y_random(x_train_scaled, x_test_scaled, y_train, y_test, metrics_cv, metrics_test, x)
        
        # Collect metrics for result_cv.csv
        result_row_cv = {
            "Feature": x,
            "Accuracy": metrics_cv.loc[x, 'Accuracy'],
            "Sensitivity": metrics_cv.loc[x, 'Sensitivity'],
            "Specificity": metrics_cv.loc[x, 'Specificity'],
            "MCC": metrics_cv.loc[x, 'MCC'],
            "AUROC": metrics_cv.loc[x, 'AUC'],
            "AUPRC": metrics_cv.loc[x, 'AUPRC']
        }
        all_results.append(result_row_cv)
        
        # Save results after each fingerprint (append mode)
        results_cv = pd.DataFrame([result_row_cv])
        results_file = "results_cv.csv"

        if os.path.exists(results_file):
            existing = pd.read_csv(results_file)
            results_cv = pd.concat([existing, results_cv], ignore_index=True)

        results_cv.to_csv(results_file, index=False)
        print(f"✅ Results CV appended for {x}")
    
        # Collect metrics for result_test.csv
        result_row = {
            "Feature": x,
            "Accuracy": metrics_test.loc[x, 'Accuracy'],
            "Sensitivity": metrics_test.loc[x, 'Sensitivity'],
            "Specificity": metrics_test.loc[x, 'Specificity'],
            "MCC": metrics_test.loc[x, 'MCC'],
            "AUROC": metrics_test.loc[x, 'AUC'],
            "AUPRC": metrics_test.loc[x, 'AUPRC']
        }
        all_results.append(result_row)

        # Save results after each fingerprint (append mode)
        results_test = pd.DataFrame([result_row])
        results_file = "results_test.csv"

        if os.path.exists(results_file):
            existing = pd.read_csv(results_file)
            results_test = pd.concat([existing, results_test], ignore_index=True)

        results_test.to_csv(results_file, index=False)
        print(f"✅ Results appended for {x}")

        # Save the Random Forest model and scaler
        model_save_path = os.path.join("models", f"{x.lower()}_model.joblib")
        os.makedirs("models", exist_ok=True)
        joblib.dump((model, scaler), model_save_path)
        print(f"\n✅ Model and scaler saved as '{model_save_path}'")

if __name__ == "__main__":
    main()

