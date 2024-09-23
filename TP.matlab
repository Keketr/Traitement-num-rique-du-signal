function [x_pred, P_pred] = EKF_Prediction_Step(delta_t, v_k, om_k, P_prev, x_prev, Q)

    % Extraction des états précédents
    x = x_prev(1); % position en x
    y = x_prev(2); % position en y
    theta = x_prev(3); % orientation

    % Mise à jour de la position prédite (modèle de mouvement)
    x_pred = x_prev + delta_t * [cos(theta), 0; sin(theta), 0; 0, 1] * [v_k; om_k];

    % Calcul de la jacobienne F_k-1
    F = [1, 0, -v_k * delta_t * sin(theta);
         0, 1,  v_k * delta_t * cos(theta);
         0, 0, 1];

    % Calcul de la jacobienne L_k-1 (lié au bruit)
    L = [cos(theta) * delta_t, 0;
         sin(theta) * delta_t, 0;
         0, delta_t];

    % Mise à jour de la covariance prédite
    P_pred = F * P_prev * F' + L * Q * L';

end


/////////////////////////////////////////////////////////////////


th_prev = wrapToPi(x_hat(3)); % Enveloppe l'angle précédent dans [-pi, pi]
vom = [vk, omk].';           % Regroupe vitesse linéaire et angulaire

Hh = [cos(th_prev), 0;
      sin(th_prev), 0;
      0,            1];       % Matrice de transformation liée à l'odométrie

x_check = x_hat + delta_t * Hh * vom; % Mise à jour de l'état
x_check(3) = wrapToPi(x_check(3));    % Enveloppe l'angle pour rester dans [-pi, pi]

F_km = zeros(3,3);  % Initialise la matrice Jacobienne
F_km(:,1) = [1 0 0].'; % Remplit la première colonne (derivée par rapport à x)
F_km(:,2) = [0 1 0].'; % Remplit la deuxième colonne (derivée par rapport à y)

F_km(1,3) = -vk * delta_t * sin(th_prev); % Terme lié à theta dans la première ligne
F_km(2,3) =  vk * delta_t * cos(th_prev); % Terme lié à theta dans la deuxième ligne
F_km(3,3) = 1; % Terme lié à theta dans la troisième ligne

L_km = zeros(3, 2);    % Initialisation de la matrice Jacobienne pour le bruit
L_km(1,1) = cos(th_prev) * delta_t; % Influence du bruit sur x
L_km(2,1) = sin(th_prev) * delta_t; % Influence du bruit sur y
L_km(3,2) = delta_t;               % Influence du bruit sur theta

P_check = F_km * P_hat * F_km' + L_km * Qkm * L_km';


