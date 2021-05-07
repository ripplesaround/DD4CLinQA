from AutoLabReport.AutoLabReport import AutoLabReport

if __name__== "__main__":
    ALR = AutoLabReport()
    ALR.get_text(text="你好")
    ALR.build_pdf()