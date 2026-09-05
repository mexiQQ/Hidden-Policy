ICLR_STYLE_URL := https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip
TEMPLATE_DIR := .iclr-template
PAPER_DIR := paper

.PHONY: all template clean

all:
	cd $(PAPER_DIR) && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

template:
	rm -rf $(TEMPLATE_DIR)
	mkdir -p $(TEMPLATE_DIR)
	curl -L $(ICLR_STYLE_URL) -o $(TEMPLATE_DIR)/iclr2027.zip
	unzip -o $(TEMPLATE_DIR)/iclr2027.zip -d $(TEMPLATE_DIR)
	find $(TEMPLATE_DIR) -name 'iclr2027_conference.sty' -exec cp {} $(PAPER_DIR)/iclr2027_conference.sty \;
	find $(TEMPLATE_DIR) -name 'iclr2027_conference.bst' -exec cp {} $(PAPER_DIR)/iclr2027_conference.bst \;
	@test -f $(PAPER_DIR)/iclr2027_conference.sty || (echo 'Could not find iclr2027_conference.sty in official archive' && exit 1)
	@test -f $(PAPER_DIR)/iclr2027_conference.bst || (echo 'Could not find iclr2027_conference.bst in official archive' && exit 1)
	rm -rf $(TEMPLATE_DIR)

clean:
	cd $(PAPER_DIR) && latexmk -C
