import os

def parse_annotation(txt_path):
    annotations=[]
    with open(txt_path,'r')as f:
        for line in f:
            if line.strip()=="":
                continue
            start,end,crackles,wheezes=line.strip().split()
            annotations.append({
                "start":float(start),
                "end":float(end),
                "crackles":int(crackles),
                "wheezes":int(wheezes)
            })
    return annotations

def get_label(crackles,wheezes):
    if crackles==0 and wheezes==0:
        return "normal"
    elif crackles==1 and wheezes==0:
        return "Crackles"
    elif crackles==0 and wheezes==1:
        return "wheezes"
    else:
        return "Both"

if __name__=="__main__":
    txt_file=r"C:\Users\akhil\Desktop\respiration\data\raw_data\101_1b1_Al_sc_Meditron.txt"
    anns=parse_annotation(txt_file)
    print("number of segmants:",len(anns))
    for a in anns[:3]:
        print(a,"->",get_label(a["crackles"],a['wheezes']))