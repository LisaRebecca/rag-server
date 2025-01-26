# Knowledge Services

The folder `knowledge-service`contains different instances of the service category called knowledge service. A `knowledge service` is an abstraction over a knowledge source. It provides a common interface to knowledge consumers. A knowledge service can be run independently, providing a search engine over the knowledge base.

```
.
├── Insomnia_2025-01-25.json    # Insomnia workspace (import to insomnia to use)
├── Makefile                    # not yet implemented
├── README.md                   # this file
├── common                      # shared files (ex. base models)
├── docker-compose.yml          # not yet implemented
├── document-knowledge-service  # not yet implemented
├── knowledge_service.egg-info  # egg file to allow absolute imports
├── mock-knowledge-service      # exemplary knowledge service working in docker + python
├── scripts                     # not yet implemented
├── setup.py                    # setup file to allow absolute imports
└── website-knowledge-service   # not yet implemented
```