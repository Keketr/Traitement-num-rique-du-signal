import argparse
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Chemin vers houses.csv")
    parser.add_argument("--target", required=True, help="Nom de la colonne cible (ex: price)")
    parser.add_argument("--out", default="linear_model.joblib", help="Nom du modèle .joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    if args.target not in df.columns:
        raise ValueError(f"Colonne cible '{args.target}' introuvable. Colonnes: {list(df.columns)}")

    # On garde uniquement les colonnes numériques
    df_num = df.select_dtypes(include=["number"]).copy()

    if args.target not in df_num.columns:
        raise ValueError("La colonne target n'est pas numérique. (TP: prends une target numérique)")

    y = df_num[args.target].values
    X = df_num.drop(columns=[args.target]).values

    if X.shape[1] == 0:
        raise ValueError("Aucune feature numérique trouvée (à part la target).")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    dump(model, args.out)

    print(f"[OK] Modèle entraîné et sauvegardé: {args.out}")
    print(f"[INFO] Nb features: {X.shape[1]}")
    print(f"[INFO] Coefs (premiers): {model.coef_[:min(5, len(model.coef_))]}")
    print(f"[INFO] Intercept: {model.intercept_}")


if __name__ == "__main__":
    main()
