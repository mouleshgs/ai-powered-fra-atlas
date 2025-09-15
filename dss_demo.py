from dss import recommend_schemes, recommend_with_priority, extract_features_for_ml

SAMPLE_APPLICANTS = [
    {
        'name': 'Rama Singh',
        'land_size': 1.2,
        'tribe': False,
        'water_access': 'well',
        'income': 45000,
        'age': 45,
        'gender': 'female',
        'household_size': 5,
        'agriculture': True,
        'labourer': False,
        'housing_deficit': True,
        'disability': False,
        'female_headed': True,
    },
    {
        'name': 'Mangal Tudu',
        'land_size': 0.4,
        'tribe': True,
        'water_access': 'none',
        'income': 30000,
        'age': 38,
        'gender': 'male',
        'household_size': 6,
        'agriculture': True,
        'labourer': False,
        'housing_deficit': False,
        'disability': False,
    },
    {
        'name': 'Sita Bai',
        'land_size': 3.5,
        'tribe': False,
        'water_access': 'piped',
        'income': 150000,
        'age': 50,
        'gender': 'female',
        'household_size': 4,
        'agriculture': False,
        'labourer': True,
        'housing_deficit': True,
        'disability': False,
    },
    {
        'name': 'Gopal Das',
        'land_size': 0.0,
        'tribe': False,
        'water_access': 'well',
        'income': 60000,
        'age': 30,
        'gender': 'male',
        'household_size': 3,
        'agriculture': False,
        'labourer': True,
        'housing_deficit': False,
        'disability': True,
    },
    {
        'name': 'Jhumpa Munda',
        'land_size': 1.0,
        'tribe': True,
        'water_access': 'none',
        'income': 20000,
        'age': 28,
        'gender': 'female',
        'household_size': 7,
        'agriculture': True,
        'labourer': False,
        'housing_deficit': True,
        'disability': False,
    }
]


def main():
    for a in SAMPLE_APPLICANTS:
        print('-' * 60)
        print('Applicant:', a['name'])
        schemes = recommend_schemes(a)
        print('Eligible schemes:', schemes)
        scored = recommend_with_priority(a)
        print('Scored recommendations:')
        for s in scored:
            print(f"  - {s['scheme']}: {s['score']} pts; reasons: {s['reasons']}")
        print('Features for ML (example):', extract_features_for_ml(a))

if __name__ == '__main__':
    main()
