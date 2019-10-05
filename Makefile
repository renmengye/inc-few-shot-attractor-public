PROTOC=protoc
proto: fewshot/configs/*.proto
	$(PROTOC) fewshot/configs/*.proto --python_out .

clean:
	rm -rf fewshot/configs/*_pb2.py
