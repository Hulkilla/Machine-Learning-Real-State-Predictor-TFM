CREATE TABLE Comodidades (
    "Terraza" VARCHAR(2),
    "Garaje" VARCHAR(2),
    "Piscina" VARCHAR(2),
    "Comodidades" TEXT
);
INSERT INTO Comodidades ("Terraza", "Garaje", "Piscina", "Comodidades")
VALUES
    ('Sí', 'Sí', 'Sí', '[TERRAZA, GARAJE, PISCINA]'),
    ('No', 'Sí', 'Sí', '[GARAJE, PISCINA]'),
    ('Sí', 'No', 'Sí', '[TERRAZA, PISCINA]'),
    ('No', 'No', 'Sí', '[PISCINA]'),
    ('Sí', 'Sí', 'No', '[TERRAZA, GARAJE]'),
    ('No', 'Sí', 'No', '[GARAJE]'),
    ('Sí', 'No', 'No', '[TERRAZA]'),
    ('No', 'No', 'No', '[]');
